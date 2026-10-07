"""
Audio Utilities for Loading, Preprocessing, Normalization, Windowing, and Feature Extraction
"""

import io
import os
import tempfile
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


def compute_file_hash(file_input: Union[str, Path, bytes, Any]) -> str:
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
    file_input: Union[str, Path, bytes, io.BytesIO, Any],
    target_sr: int = config.SAMPLE_RATE,
    max_duration: float = config.MAX_AUDIO_SECONDS,
    mono: bool = config.MONO
) -> Tuple[np.ndarray, int, float]:
    """
    Loads audio from a file path, byte stream, or file-like object, converts to mono, resamples to target_sr (16 kHz),
    checks for invalid/corrupted audio, normalizes amplitude, and caps at max_duration (60s).

    Returns:
        waveform (np.ndarray): 1D float32 array in range [-1.0, 1.0]
        sample_rate (int): target sample rate (16000 Hz)
        duration (float): duration in seconds before truncation
    """
    waveform = None
    orig_sr = None
    audio_bytes = None
    file_path = None

    # Determine input type & extract raw bytes / path
    if isinstance(file_input, (str, Path)):
        file_path = Path(file_input)
    elif isinstance(file_input, bytes):
        audio_bytes = file_input
    elif hasattr(file_input, "read"):
        audio_bytes = file_input.read()
        if hasattr(file_input, "seek"):
            file_input.seek(0)

    # 1. Soundfile stream / path attempt
    try:
        if file_path is not None:
            data, sr_read = sf.read(str(file_path), dtype="float32")
        elif audio_bytes is not None:
            data, sr_read = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        else:
            data, sr_read = sf.read(file_input, dtype="float32")

        orig_sr = sr_read
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if orig_sr != target_sr:
            num = target_sr
            den = orig_sr
            gcd = np.gcd(num, den)
            data = signal.resample_poly(data, num // gcd, den // gcd)

        waveform = data
    except Exception:
        waveform = None

    # 2. Librosa stream / path attempt
    if waveform is None and librosa is not None:
        try:
            if file_path is not None:
                waveform, _ = librosa.load(str(file_path), sr=target_sr, mono=mono)
            elif audio_bytes is not None:
                waveform, _ = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=mono)
            else:
                waveform, _ = librosa.load(file_input, sr=target_sr, mono=mono)
        except Exception:
            waveform = None

    # 3. Disk fallback via NamedTemporaryFile if stream decoding failed
    if waveform is None and audio_bytes is not None:
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_file = tmp.name

            # Try soundfile then librosa on temp file
            try:
                data, sr_read = sf.read(tmp_file, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr_read != target_sr:
                    num = target_sr
                    den = sr_read
                    gcd = np.gcd(num, den)
                    data = signal.resample_poly(data, num // gcd, den // gcd)
                waveform = data
            except Exception:
                if librosa is not None:
                    waveform, _ = librosa.load(tmp_file, sr=target_sr, mono=mono)
        except Exception as e:
            raise ValueError(f"Failed to decode audio file: {str(e)}")
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except Exception:
                    pass

    if waveform is None or len(waveform) == 0:
        raise ValueError("Audio file is empty or unreadable.")

    # Clean non-finite numbers (NaN/Inf)
    if not np.isfinite(waveform).all():
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    original_duration = len(waveform) / target_sr

    # Measure raw max amplitude before peak normalization
    raw_max_amp = float(np.max(np.abs(waveform)))

    # Only peak-normalize if raw signal is above noise floor (>= 1e-4)
    if raw_max_amp >= 1e-4:
        waveform = waveform / raw_max_amp
    else:
        waveform = waveform.copy()

    # Truncate if longer than max_duration (60 seconds = 960,000 samples at 16 kHz)
    max_samples = int(max_duration * target_sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    return waveform.astype(np.float32), target_sr, original_duration


def check_audio_quality(
    waveform: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    original_duration: float = 0.0
) -> Dict[str, Any]:
    """
    Deterministically validates audio usability and quality BEFORE machine learning model inference.

    Inspects:
    1. Corrupted, empty, or non-finite audio arrays
    2. Completely silent or near-silent audio (RMS level or peak amplitude < threshold)
    3. Insufficient speech/voice activity duration (< 0.35 seconds)
    4. No speech / non-speech frequency hum / sub-bass noise
    5. Heavy broadband noise / extreme disturbance (spectral flatness > 0.60)

    Returns:
        dict containing:
            - is_usable (bool): True if audio passes quality validation, False if rejected
            - status (str): "PASSED" or "FAILED"
            - reason (str): Human-readable rejection reason (if FAILED), else None
            - metrics (dict): Internal telemetry (duration, sample_rate, channels, rms_level, non_silent_percentage, etc.)
    """
    metrics = {
        "duration": float(original_duration),
        "sample_rate": int(sr),
        "channels": 1,
        "rms_level": 0.0,
        "non_silent_percentage": 0.0,
        "active_duration_sec": 0.0,
        "spectral_flatness": 0.0,
        "speech_band_ratio": 0.0,
        "quality_status": "PASSED",
        "quality_rejection_reason": None
    }

    # Gate 1: Corrupted or invalid array
    if waveform is None or len(waveform) == 0 or not np.isfinite(waveform).all():
        metrics["quality_status"] = "FAILED"
        metrics["quality_rejection_reason"] = "Audio file is corrupted or unreadable."
        return {
            "is_usable": False,
            "status": "FAILED",
            "reason": "Audio file is corrupted or unreadable.",
            "metrics": metrics
        }

    rms_val = float(np.sqrt(np.mean(waveform**2)))
    max_amp = float(np.max(np.abs(waveform)))
    metrics["rms_level"] = round(rms_val, 6)

    # Gate 2: Completely Muted or Near-Silent Audio (Digital zero / noise floor < 1e-4)
    if max_amp < 1e-4 or rms_val < 1e-5:
        metrics["quality_status"] = "FAILED"
        metrics["quality_rejection_reason"] = "No usable audio signal was detected."
        return {
            "is_usable": False,
            "status": "FAILED",
            "reason": "No usable audio signal was detected.",
            "metrics": metrics
        }

    # Frame-level Voice Activity Detection (VAD)
    frame_len = int(0.030 * sr)
    hop_len = int(0.015 * sr)
    num_frames = max(1, (len(waveform) - frame_len) // hop_len + 1)
    
    frame_rms = []
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        chunk = waveform[start:end]
        f_rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
        frame_rms.append(f_rms)
        
    frame_rms = np.array(frame_rms)
    
    # Energy threshold for active speech in normalized waveform
    energy_thresh = getattr(config, "QUALITY_ENERGY_THRESHOLD", 0.015)
    non_silent_mask = frame_rms > energy_thresh
    non_silent_count = int(np.sum(non_silent_mask))
    non_silent_pct = float((non_silent_count / len(frame_rms)) * 100.0)
    active_duration = float(non_silent_count * (hop_len / sr))
    
    metrics["non_silent_percentage"] = round(non_silent_pct, 2)
    metrics["active_duration_sec"] = round(active_duration, 2)

    # Gate 3: Very Low Energy / Near Silent Signal
    if rms_val < 0.003 or (non_silent_pct < 2.0 and active_duration < 0.2):
        metrics["quality_status"] = "FAILED"
        metrics["quality_rejection_reason"] = "No usable audio signal was detected."
        return {
            "is_usable": False,
            "status": "FAILED",
            "reason": "No usable audio signal was detected.",
            "metrics": metrics
        }

    # Gate 4 & 5: Frequency Energy & Spectral Flatness Check
    try:
        n_fft = 512
        _, _, Zxx = signal.stft(waveform, fs=sr, nperseg=n_fft)
        mag_spec = np.abs(Zxx) + 1e-10
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        
        speech_band_mask = (freqs >= 300) & (freqs <= 3400)
        total_energy = np.sum(mag_spec**2)
        speech_energy = np.sum(mag_spec[speech_band_mask, :]**2)
        speech_band_ratio = float(speech_energy / total_energy) if total_energy > 0 else 0.0
        metrics["speech_band_ratio"] = round(speech_band_ratio, 4)

        # Gate 4: Insufficient Speech / No Voice Content
        min_speech_dur = getattr(config, "QUALITY_MIN_SPEECH_DURATION", 0.35)
        if speech_band_ratio < 0.15 or active_duration < min_speech_dur:
            metrics["quality_status"] = "FAILED"
            metrics["quality_rejection_reason"] = "Insufficient speech content for reliable forensic analysis."
            return {
                "is_usable": False,
                "status": "FAILED",
                "reason": "Insufficient speech content for reliable forensic analysis.",
                "metrics": metrics
            }

        # Gate 5: Heavy Disturbance / Extreme Noise Check (Spectral Flatness)
        spec_mean = np.mean(mag_spec, axis=1)
        log_spec = np.log(spec_mean)
        gmean = np.exp(np.mean(log_spec))
        amean = np.mean(spec_mean)
        spec_flatness = float(gmean / amean)
        metrics["spectral_flatness"] = round(spec_flatness, 4)

        max_flatness = getattr(config, "QUALITY_MAX_SPECTRAL_FLATNESS", 0.70)
        # Flat broadband noise has flatness > 0.60
        if spec_flatness > 0.60:
            metrics["quality_status"] = "FAILED"
            metrics["quality_rejection_reason"] = "Audio quality is insufficient for reliable forensic analysis."
            return {
                "is_usable": False,
                "status": "FAILED",
                "reason": "Audio quality is insufficient for reliable forensic analysis.",
                "metrics": metrics
            }
    except Exception:
        pass

    metrics["quality_status"] = "PASSED"
    return {
        "is_usable": True,
        "status": "PASSED",
        "reason": None,
        "metrics": metrics
    }





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
    For audio > 4s, end-aligns tail windows to eliminate artificial zero-padding artifacts.

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
    while start_idx + window_samples <= total_samples:
        end_idx = start_idx + window_samples
        chunk = waveform[start_idx:end_idx]
        start_sec = round(start_idx / target_sr, 2)
        end_sec = round(end_idx / target_sr, 2)
        windows.append((chunk, start_sec, end_sec))
        start_idx += hop_samples

    # If remaining un-analyzed tail is at least 0.5s, add an end-aligned window (no zero-padding)
    if start_idx < total_samples and (total_samples - start_idx) >= int(0.5 * target_sr):
        tail_start = total_samples - window_samples
        if tail_start > 0 and (not windows or tail_start > int(windows[-1][0].shape[0])):
            chunk = waveform[tail_start:total_samples]
            start_sec = round(tail_start / target_sr, 2)
            end_sec = round(total_samples / target_sr, 2)
            windows.append((chunk, start_sec, end_sec))

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


