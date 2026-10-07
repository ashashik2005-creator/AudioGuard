"""
Command-Line Interface (CLI) Prediction Tool for Audio Deepfake Detection
Usage: python predict.py --file sample.wav
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Union, Tuple

import numpy as np
import joblib

import config
from audio_utils import load_and_preprocess_audio, extract_sliding_windows, check_audio_quality
from features import HandcraftedFeatureExtractor


def load_trained_model_and_config() -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads saved model weights, feature extractor, and configuration parameters.
    """
    config_file = config.MODELS_DIR / "config.json"
    if not config_file.exists():
        raise FileNotFoundError("No trained model config found in models/config.json. Please run train.py first.")

    with open(config_file, "r") as f:
        config_info = json.load(f)

    model_path = config.MODELS_DIR / "mfcc_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"MFCC classifier model not found at {model_path}")

    pipeline = joblib.load(model_path)
    extractor = HandcraftedFeatureExtractor()

    return pipeline, extractor, config_info


def predict_audio_file(audio_path: Union[str, Path, bytes, Any], backend: str = None) -> Dict[str, Any]:
    """
    Runs audio deepfake inference on an input audio file path, byte stream, or file object.

    First validates audio quality and usability BEFORE the ML pipeline.
    If usable, proceeds to existing MFCC + StandardScaler + RandomForest model pipeline.
    If unusable, returns INCONCLUSIVE without executing model inference.
    """
    start_time = time.time()
    file_label = "uploaded_audio.wav"

    if isinstance(audio_path, (str, Path)):
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")
        file_label = str(path_obj)
    elif hasattr(audio_path, "name"):
        file_label = str(getattr(audio_path, "name"))

    # Load model and config
    classifier, extractor, config_info = load_trained_model_and_config()
    selected_backend = config_info.get("backend", "mfcc")
    threshold = config_info.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD)
    uncertainty_margin = config_info.get("uncertainty_margin", config.UNCERTAINTY_MARGIN)

    # 1. Preprocess audio & run deterministic quality check
    try:
        waveform, sr, original_duration = load_and_preprocess_audio(audio_path)
        quality_res = check_audio_quality(waveform, sr=sr, original_duration=original_duration)
    except Exception as e:
        quality_res = {
            "is_usable": False,
            "status": "FAILED",
            "reason": "Audio file is corrupted or unreadable.",
            "metrics": {
                "duration": 0.0,
                "sample_rate": config.SAMPLE_RATE,
                "channels": 1,
                "rms_level": 0.0,
                "non_silent_percentage": 0.0,
                "active_duration_sec": 0.0,
                "spectral_flatness": 0.0,
                "quality_status": "FAILED",
                "quality_rejection_reason": "Audio file is corrupted or unreadable."
            }
        }
        waveform = np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
        sr = config.SAMPLE_RATE
        original_duration = 0.0

    # 2. AUDIO QUALITY GATE: If unusable, BYPASS ML model inference entirely
    if not quality_res["is_usable"]:
        processing_time = round(time.time() - start_time, 2)
        return {
            "file": file_label,
            "prediction": "INCONCLUSIVE",
            "fake_probability": 50.0,
            "real_probability": 50.0,
            "confidence": 50.0,
            "decision_threshold": threshold,
            "uncertainty_margin": uncertainty_margin,
            "is_uncertain": True,
            "is_quality_rejected": True,
            "quality_status": "FAILED",
            "quality_rejection_reason": quality_res["reason"],
            "processing_time_sec": processing_time,
            "backend_used": selected_backend,
            "audio_duration_sec": round(original_duration, 2),
            "quality_telemetry": quality_res["metrics"],
            "windows": []
        }

    # 3. EXISTING ML PIPELINE: Run sliding windows through MFCC + StandardScaler + RandomForest
    windows = extract_sliding_windows(waveform, target_sr=sr)

    window_results = []
    fake_probs = []

    for win_wave, start_sec, end_sec in windows:
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

    lower_bound = threshold - uncertainty_margin  # 0.50 - 0.05 = 0.45
    upper_bound = threshold + uncertainty_margin  # 0.50 + 0.05 = 0.55

    # Threshold evaluation on actual unrounded probability
    if overall_fake_prob > upper_bound:
        prediction_label = "AI-GENERATED"
        is_uncertain = False
        confidence_pct = overall_fake_prob * 100.0
    elif overall_fake_prob < lower_bound:
        prediction_label = "REAL"
        is_uncertain = False
        confidence_pct = overall_real_prob * 100.0
    else:
        prediction_label = "INCONCLUSIVE"
        is_uncertain = True
        confidence_pct = max(overall_fake_prob, overall_real_prob) * 100.0

    result = {
        "file": file_label,
        "prediction": prediction_label,
        "fake_probability": round(overall_fake_prob * 100, 2),
        "real_probability": round(overall_real_prob * 100, 2),
        "confidence": round(confidence_pct, 2),
        "decision_threshold": threshold,
        "uncertainty_margin": uncertainty_margin,
        "is_uncertain": is_uncertain,
        "is_quality_rejected": False,
        "quality_status": "PASSED",
        "quality_rejection_reason": None,
        "processing_time_sec": processing_time,
        "backend_used": selected_backend,
        "audio_duration_sec": round(original_duration, 2),
        "quality_telemetry": quality_res["metrics"],
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

    if result.get("is_quality_rejected"):
        print("\nPrediction: INCONCLUSIVE")
        print(f"\nQuality Status: FAILED")
        print(f"Reason:         {result.get('quality_rejection_reason')}")
        print(f"\nProcessing time: {result['processing_time_sec']:.2f} seconds")
        print("\nResult is inconclusive because the audio quality is insufficient for reliable forensic analysis.")
    elif result["is_uncertain"] or result["prediction"] == "INCONCLUSIVE":
        print("\nPrediction: INCONCLUSIVE")
        print(f"Fake probability: {result['fake_probability']:.2f}%")
        print(f"Real probability: {result['real_probability']:.2f}%")
        print(f"Confidence:       {result['confidence']:.2f}%")
        print(f"\nProcessing time:  {result['processing_time_sec']:.2f} seconds")
        print("\nResult is inconclusive.")
        print("Model probabilities fall within the uncertainty range (45% - 55%).")
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
    args = parser.parse_args()

    res = predict_audio_file(args.file)
    print_cli_prediction(res)
