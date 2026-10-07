"""
Audio Utilities for Loading, Preprocessing, Normalization, Windowing, and Feature Extraction
"""

import io
import hashlib
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.fft as fft

import config

try:
    import librosa
except Exception:
    librosa = None


def compute_file_hash(file_input: Union[str, Path, bytes]) -> str:
    """
    Computes SHA-256 hash for a file path or bytes object to detect exact duplicates.
    """
    hasher = hashlib.sha256()
    if isinstance(file_input, (str, Path)):
        with open(file_input, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    elif isinstance(file_input, bytes):
        hasher.update(file_input)
    elif hasattr(file_input, "read"):
        data = file_input.read()
        hasher.update(data)
        if hasattr(file_input, "seek"):
            file_input.seek(0)
    else:
        raise ValueError("Unsupported input type for hash computation.")
    return hasher.hexdigest()


def load_and_preprocess_audio(
    file_input: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = config.SAMPLE_RATE,
    max_duration: float = config.MAX_AUDIO_SECONDS,
    mono: bool = config.MONO
) -> Tuple[np.ndarray, int, float]:
    """
    Loads audio from a file path or byte stream, converts to mono, resamples to target_sr (16 kHz),
    checks for invalid/corrupted audio, normalizes amplitude, and caps at max_duration (60s).

    Returns:
        waveform (np.ndarray): 1D float32 array in range [-1.0, 1.0]
        sample_rate (int): target sample rate (16000 Hz)
        duration (float): duration in seconds before truncation
    """
    waveform = None
    orig_sr = None

    # 1. Try soundfile decoding first (fast, C-based, no Numba dependency)
    try:
        if isinstance(file_input, bytes):
            bio = io.BytesIO(file_input)
        elif isinstance(file_input, io.BytesIO):
            bio = file_input
            bio.seek(0)
        else:
            bio = file_input

        data, sr_read = sf.read(bio, dtype="float32")
        orig_sr = sr_read

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if orig_sr != target_sr:
            # Resample via scipy.signal.resample_poly
            num = target_sr
            den = orig_sr
            gcd = np.gcd(num, den)
            data = signal.resample_poly(data, num // gcd, den // gcd)

        waveform = data
    except Exception:
        waveform = None

    # 2. Fallback to librosa if soundfile failed and librosa is available
    if waveform is None and librosa is not None:
        try:
            if isinstance(file_input, bytes):
                file_input = io.BytesIO(file_input)
            elif isinstance(file_input, io.BytesIO):
                file_input.seek(0)

            waveform, _ = librosa.load(file_input, sr=target_sr, mono=mono)
        except Exception as e:
            raise ValueError(f"Failed to decode audio file: {str(e)}")

    if waveform is None or len(waveform) == 0:
        raise ValueError("Audio file is empty or unreadable.")

    # Clean non-finite numbers (NaN/Inf)
    if not np.isfinite(waveform).all():
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    original_duration = len(waveform) / target_sr

    # Check for silent / zero audio
    max_amp = np.max(np.abs(waveform))
    if max_amp < 1e-6:
        raise ValueError("Audio waveform is silent or contains no audible signal.")

    # Peak normalization
    waveform = waveform / max_amp

    # Truncate if longer than max_duration (60 seconds = 960,000 samples at 16 kHz)
    max_samples = int(max_duration * target_sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    return waveform.astype(np.float32), target_sr, original_duration


def pad_or_crop_waveform(
    waveform: np.ndarray,
    target_samples: int = config.WINDOW_SAMPLES
) -> np.ndarray:
    """
    Pads audio shorter than target_samples (4s = 64000 samples) with zeros,
    or crops audio longer than target_samples.
    """
    if len(waveform) == target_samples:
        return waveform
    elif len(waveform) < target_samples:
        pad_width = target_samples - len(waveform)
        return np.pad(waveform, (0, pad_width), mode="constant", constant_values=0.0)
    else:
        return waveform[:target_samples]


def extract_sliding_windows(
    waveform: np.ndarray,
    window_samples: int = config.WINDOW_SAMPLES,
    hop_samples: int = config.HOP_SAMPLES,
    target_sr: int = config.SAMPLE_RATE
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Divides an audio waveform into 4-second overlapping windows with 2-second hop step.

    Returns:
        List of tuples: (window_waveform, start_time_sec, end_time_sec)
    """
    total_samples = len(waveform)

    # If audio is shorter than or equal to window_samples, return single padded window
    if total_samples <= window_samples:
        padded = pad_or_crop_waveform(waveform, window_samples)
        duration_sec = total_samples / target_sr
        return [(padded, 0.0, duration_sec)]

    windows = []
    start_idx = 0
    while start_idx < total_samples:
        end_idx = start_idx + window_samples
        chunk = waveform[start_idx:min(end_idx, total_samples)]

        # If last window is shorter than window_samples, pad it
        if len(chunk) < window_samples:
            chunk = pad_or_crop_waveform(chunk, window_samples)

        start_sec = round(start_idx / target_sr, 2)
        end_sec = round(min(end_idx, total_samples) / target_sr, 2)

        windows.append((chunk, start_sec, end_sec))

        # Move hop step
        start_idx += hop_samples

        # Break if remaining samples are insignificant (less than 0.5 sec)
        if total_samples - start_idx < int(0.5 * target_sr):
            break

    return windows


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)


def _get_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmin, fmax = 0.0, sr / 2.0
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i-1], bins[i], bins[i+1]
        if center > left:
            fb[i-1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i-1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64
) -> np.ndarray:
    """
    Computes Log Mel-Spectrogram in dB scale for audio visualization.
    """
    if librosa is not None:
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            return librosa.power_to_db(mel_spec, ref=np.max)
        except Exception:
            pass

    # SciPy fallback
    f, t, Zxx = signal.stft(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    mag = np.abs(Zxx) + 1e-10
    fb = _get_mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(fb, mag)
    mel_spec_db = 10 * np.log10(np.maximum(1e-10, mel_spec))
    return mel_spec_db - np.max(mel_spec_db)


def compute_mfcc_visualization(
    waveform: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = 20
) -> np.ndarray:
    """
    Computes MFCC matrix for visualization.
    """
    if librosa is not None:
        try:
            return librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
        except Exception:
            pass

    # SciPy fallback
    n_fft = 1024
    hop_length = 512
    f, t, Zxx = signal.stft(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    mag = np.abs(Zxx) + 1e-10
    fb = _get_mel_filterbank(sr, n_fft, 40)
    mel_spec = np.dot(fb, mag)
    log_mel = np.log(np.maximum(1e-10, mel_spec))
    mfcc = fft.dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return mfcc


def apply_mild_training_augmentation(waveform: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Applies mild, non-distorting audio augmentation exclusively for training set generalization:
    - Mild Gaussian background noise (SNR ~35 dB)
    - Mild gain/volume scaling (0.85 - 1.15)
    - Mild time shift (up to 50 ms)
    """
    aug_wave = waveform.copy()
    
    # 1. Mild Gain Scaling
    gain = np.random.uniform(0.85, 1.15)
    aug_wave = aug_wave * gain
    
    # 2. Mild Time Shift
    max_shift = int(sr * 0.05)  # 50 ms
    shift = np.random.randint(-max_shift, max_shift)
    aug_wave = np.roll(aug_wave, shift)
    
    # 3. Mild Background Noise (SNR 35dB)
    signal_power = np.mean(aug_wave**2) + 1e-10
    noise_power = signal_power / (10 ** (35 / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(aug_wave))
    aug_wave = aug_wave + noise
    
    # Re-normalize peak amplitude
    max_val = np.max(np.abs(aug_wave))
    if max_val > 0:
        aug_wave = aug_wave / max_val
        
    return aug_wave.astype(np.float32)


