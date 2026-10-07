"""
AudioGuard — External Audio Evaluation Utility
Tests external audio (friend's voice, phone recording, laptop mic, WhatsApp audio)
WITHOUT training on it or modifying the ML pipeline.

Pipeline executed:
Audio -> Preprocessing -> MFCC (172-D) -> StandardScaler -> RandomForestClassifier -> Verdict
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append("e:/deepfake_detection")

import config
from predict import predict_audio_file, load_trained_model_and_config


def evaluate_external_file(file_path: Path) -> Dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"External audio file not found: {file_path}")

    res = predict_audio_file(file_path)
    fake_p = res["fake_probability"]
    real_p = res["real_probability"]
    thr = res.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD)
    margin = res.get("uncertainty_margin", config.UNCERTAINTY_MARGIN)

    lower_b = (thr - margin) * 100
    upper_b = (thr + margin) * 100

    if lower_b <= fake_p <= upper_b:
        verdict = "INCONCLUSIVE"
        desc = "Borderline result — model probabilities fall inside the uncertainty margin (45% - 55%)."
    elif fake_p < lower_b:
        verdict = "REAL"
        desc = "The model found stronger evidence consistent with authentic human speech."
    else:
        verdict = "AI-GENERATED"
        desc = "The model found stronger evidence consistent with synthetic AI speech."

    print("\n" + "=" * 55)
    print("   AUDIOGUARD EXTERNAL AUDIO FORENSIC ASSESSMENT")
    print("=" * 55)
    print(f"File Path:            {file_path}")
    print(f"Audio Duration:       {res['audio_duration_sec']:.2f} seconds")
    print(f"Decision Threshold:   {thr*100:.0f}% (Uncertainty Range: {lower_b:.0f}%–{upper_b:.0f}%)")
    print("-" * 55)
    print(f"REAL Probability:     {real_p:.2f}%")
    print(f"AI Probability:       {fake_p:.2f}%")
    print("-" * 55)
    print(f"FINAL VERDICT:        {verdict}")
    print(f"Assessment Context:   {desc}")
    print("=" * 55 + "\n")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate external audio on AudioGuard pipeline")
    parser.add_argument("--file", type=str, required=True, help="Path to external audio file (WAV, MP3, FLAC, OGG, M4A)")
    args = parser.parse_args()

    evaluate_external_file(Path(args.file))
