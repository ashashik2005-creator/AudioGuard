"""
Command-Line Interface (CLI) Prediction Tool for Audio Deepfake Detection
Usage: python predict.py --file sample.wav [--backend wav2vec|mfcc]
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Union, Tuple

import numpy as np
import torch
import joblib

import config
from audio_utils import load_and_preprocess_audio, extract_sliding_windows
from features import Wav2VecExtractor, HandcraftedFeatureExtractor
from model import Wav2VecClassifierHead


def load_trained_model_and_config(backend: str = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads saved model weights, feature extractor, and configuration parameters.
    """
    config_file = config.MODELS_DIR / "config.json"
    if not config_file.exists():
        raise FileNotFoundError("No trained model config found in models/config.json. Please run train.py first.")

    with open(config_file, "r") as f:
        config_info = json.load(f)

    selected_backend = backend.lower() if backend else config_info.get("backend", "wav2vec")

    if selected_backend == "wav2vec":
        model_path = config.MODELS_DIR / "wav2vec_classifier.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Wav2Vec classifier weights not found at {model_path}")

        extractor = Wav2VecExtractor()
        classifier_head = Wav2VecClassifierHead(input_dim=config.WAV2VEC_EMBEDDING_DIM)
        classifier_head.load_state_dict(torch.load(model_path, map_location="cpu"))
        classifier_head.eval()

        return classifier_head, extractor, config_info

    elif selected_backend == "mfcc":
        model_path = config.MODELS_DIR / "mfcc_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"MFCC classifier model not found at {model_path}")

        pipeline = joblib.load(model_path)
        extractor = HandcraftedFeatureExtractor()

        return pipeline, extractor, config_info
    else:
        raise ValueError(f"Unknown model backend: {selected_backend}")


def predict_audio_file(audio_path: Union[str, Path], backend: str = None) -> Dict[str, Any]:
    """
    Runs audio deepfake inference on an input audio file.

    Returns prediction summary containing label, probabilities, confidence,
    processing time, and window-level breakdown.
    """
    start_time = time.time()
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load model and config
    classifier, extractor, config_info = load_trained_model_and_config(backend=backend)
    selected_backend = backend.lower() if backend else config_info.get("backend", "wav2vec")
    threshold = config_info.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD)
    uncertainty_margin = config_info.get("uncertainty_margin", config.UNCERTAINTY_MARGIN)

    # Preprocess audio
    waveform, sr, original_duration = load_and_preprocess_audio(audio_path)
    windows = extract_sliding_windows(waveform, target_sr=sr)

    window_results = []
    fake_probs = []

    for win_wave, start_sec, end_sec in windows:
        if selected_backend == "wav2vec":
            emb = extractor.extract_window_embedding(win_wave)
            emb_tensor = torch.tensor(emb).unsqueeze(0)  # [1, 768]
            with torch.no_grad():
                prob_fake = float(classifier.predict_proba(emb_tensor).item())
        else:
            feat = extractor.extract_waveform_features(win_wave, sr)
            feat_arr = np.array(feat).reshape(1, -1)
            prob_fake = float(classifier.predict_proba(feat_arr)[0, 1])

        prob_real = 1.0 - prob_fake
        win_label = "FAKE" if prob_fake >= threshold else "REAL"

        fake_probs.append(prob_fake)
        window_results.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": win_label,
            "fake_prob": round(prob_fake * 100, 2),
            "real_prob": round(prob_real * 100, 2)
        })

    overall_fake_prob = float(np.mean(fake_probs))
    overall_real_prob = 1.0 - overall_fake_prob
    processing_time = round(time.time() - start_time, 2)

    is_fake = overall_fake_prob >= threshold
    prediction_label = "FAKE" if is_fake else "REAL"
    confidence_pct = overall_fake_prob * 100 if is_fake else overall_real_prob * 100

    # Uncertainty check
    is_uncertain = abs(overall_fake_prob - threshold) <= uncertainty_margin

    result = {
        "file": str(audio_path),
        "prediction": prediction_label,
        "fake_probability": round(overall_fake_prob * 100, 2),
        "real_probability": round(overall_real_prob * 100, 2),
        "confidence": round(confidence_pct, 2),
        "decision_threshold": threshold,
        "is_uncertain": is_uncertain,
        "processing_time_sec": processing_time,
        "backend_used": selected_backend,
        "audio_duration_sec": round(original_duration, 2),
        "windows": window_results
    }

    return result


def print_cli_prediction(result: Dict[str, Any]):
    """
    Prints cleanly formatted CLI prediction summary matching required output format.
    """
    print("\n" + "=" * 50)
    print(f"Audio file: {result['file']}")
    print("-" * 50)

    if result["is_uncertain"]:
        print("\nPrediction: UNCERTAIN")
        print(f"Fake probability: {result['fake_probability']:.2f}%")
        print(f"Real probability: {result['real_probability']:.2f}%")
        print(f"Confidence:       {result['confidence']:.2f}%")
        print(f"\nProcessing time:  {result['processing_time_sec']:.2f} seconds")
        print("\nResult is uncertain.")
        print("The audio should be treated as inconclusive.")
    else:
        print(f"\nPrediction: {result['prediction']}")
        print(f"\nFake probability: {result['fake_probability']:.2f}%")
        print(f"Real probability: {result['real_probability']:.2f}%")
        print(f"\nConfidence:       {result['confidence']:.2f}%")
        print(f"\nProcessing time:  {result['processing_time_sec']:.2f} seconds")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Audio Deepfake Prediction")
    parser.add_argument("--file", type=str, required=True, help="Path to input audio file")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["wav2vec", "mfcc"],
        help="Optional model backend override ('wav2vec' or 'mfcc')"
    )
    args = parser.parse_args()

    res = predict_audio_file(args.file, backend=args.backend)
    print_cli_prediction(res)
