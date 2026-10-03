"""
Training Script for Audio Deepfake Detection Model (MFCC + Random Forest)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import joblib
from sklearn.metrics import f1_score

import config
from features import HandcraftedFeatureExtractor
from model import build_mfcc_classifier
from evaluate import evaluate_model_performance, calculate_metrics
from download_dataset import download_hf_dataset


def train_mfcc_backend(
    classifier_type: str = "rf",
    force_reextract: bool = False
) -> Dict[str, Any]:
    """
    Trains Audio Deepfake Model using Handcrafted Audio Features (MFCC, Mel Spec, Spectral Stats).
    """
    print("\n" + "=" * 50)
    print("   TRAINING HANDCRAFTED / MFCC CLASSIFIER")
    print("=" * 50)

    extractor = HandcraftedFeatureExtractor()

    train_cache = config.FEATURES_DIR / "mfcc_train.joblib"
    valid_cache = config.FEATURES_DIR / "mfcc_valid.joblib"
    test_cache = config.FEATURES_DIR / "mfcc_test.joblib"

    X_train, y_train, _ = extractor.cache_split_features(config.TRAIN_DIR, train_cache, force=force_reextract)
    X_valid, y_valid, _ = extractor.cache_split_features(config.VALID_DIR, valid_cache, force=force_reextract)
    X_test, y_test, _ = extractor.cache_split_features(config.TEST_DIR, test_cache, force=force_reextract)

    print(f"\nFeature Shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Valid: {X_valid.shape}")
    print(f"  Test:  {X_test.shape}")

    # Build and fit Scikit-Learn pipeline
    pipeline = build_mfcc_classifier(classifier_type=classifier_type)
    print("\nFitting Scikit-Learn classifier...")
    pipeline.fit(X_train, y_train)

    # Tune threshold on Validation Set
    val_probs = pipeline.predict_proba(X_valid)[:, 1]
    best_threshold = 0.50
    best_f1 = 0.0
    for thresh in np.arange(0.30, 0.71, 0.02):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(thresh), 2)

    print(f"Tuned Decision Threshold on Validation Set: {best_threshold:.2f} (Val F1: {best_f1:.4f})")

    # Save model and config
    model_save_path = config.MODELS_DIR / "mfcc_model.joblib"
    joblib.dump(pipeline, model_save_path)

    config_info = {
        "backend": "mfcc",
        "classifier_type": classifier_type,
        "feature_dim": X_train.shape[1],
        "decision_threshold": best_threshold,
        "uncertainty_margin": config.UNCERTAINTY_MARGIN,
        "sample_rate": config.SAMPLE_RATE,
        "window_seconds": config.WINDOW_SECONDS
    }
    with open(config.MODELS_DIR / "config.json", "w") as f:
        json.dump(config_info, f, indent=4)

    print(f"Model saved to: {model_save_path}")

    # Evaluate on TEST Set
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model_performance(
        y_true=y_test,
        y_prob=test_probs,
        threshold=best_threshold,
        save_results=True,
        output_dir=config.RESULTS_DIR
    )

    print("\nFINAL TEST PERFORMANCE METRICS (MFCC Model)")
    print("=" * 45)
    print(f"  Test Accuracy:   {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision:  {test_metrics['precision']*100:.2f}%")
    print(f"  Test Recall:     {test_metrics['recall']*100:.2f}%")
    print(f"  Test F1-Score:   {test_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print("=" * 45)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detection Model")
    parser.add_argument(
        "--backend",
        type=str,
        default="mfcc",
        choices=["mfcc"],
        help="Model backend choice: 'mfcc' (default)"
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Force feature extraction recomputation"
    )
    args = parser.parse_args()

    # Ensure dataset exists, else trigger automatic download
    if not (config.TRAIN_DIR / "real").exists() or len(list((config.TRAIN_DIR / "real").glob("*.*"))) == 0:
        print("Dataset splits missing in data/processed. Downloading default dataset...")
        download_hf_dataset(max_per_class=100)

    train_mfcc_backend(
        force_reextract=args.force_reextract
    )

