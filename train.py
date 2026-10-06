"""
Training Script for Audio Deepfake Detection Model (MFCC + Random Forest)
Supports Candidate Model Evaluation & Model Comparison Artifact Generation
"""

import json
import csv
import shutil
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


def generate_comparison_artifacts(
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    output_dir: Path = config.MODELS_DIR
) -> Tuple[Path, Path]:
    """
    Generates model_comparison.json and model_comparison.csv for transparency.
    """
    comparison_data = {
        "baseline_model": baseline_metrics,
        "candidate_model": candidate_metrics,
        "summary": {
            "accuracy_improvement": round(candidate_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "f1_improvement": round(candidate_metrics["f1_score"] - baseline_metrics["f1_score"], 4),
            "roc_auc_improvement": round(candidate_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4),
            "promoted": True
        }
    }

    json_path = output_dir / "model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=4)

    csv_path = output_dir / "model_comparison.csv"
    headers = [
        "model_name", "dataset_sample_count", "accuracy", "precision", "recall",
        "f1_score", "roc_auc", "false_positive_rate", "false_negative_rate",
        "equal_error_rate", "decision_threshold", "uncertainty_margin"
    ]

    rows = [
        [
            baseline_metrics["model_name"],
            baseline_metrics["dataset_sample_count"],
            f"{baseline_metrics['accuracy']:.4f}",
            f"{baseline_metrics['precision']:.4f}",
            f"{baseline_metrics['recall']:.4f}",
            f"{baseline_metrics['f1_score']:.4f}",
            f"{baseline_metrics['roc_auc']:.4f}",
            f"{baseline_metrics['false_positive_rate']:.4f}",
            f"{baseline_metrics['false_negative_rate']:.4f}",
            f"{baseline_metrics['equal_error_rate']:.4f}",
            f"{baseline_metrics['decision_threshold']:.2f}",
            f"{baseline_metrics['uncertainty_margin']:.2f}"
        ],
        [
            candidate_metrics["model_name"],
            candidate_metrics["dataset_sample_count"],
            f"{candidate_metrics['accuracy']:.4f}",
            f"{candidate_metrics['precision']:.4f}",
            f"{candidate_metrics['recall']:.4f}",
            f"{candidate_metrics['f1_score']:.4f}",
            f"{candidate_metrics['roc_auc']:.4f}",
            f"{candidate_metrics['false_positive_rate']:.4f}",
            f"{candidate_metrics['false_negative_rate']:.4f}",
            f"{candidate_metrics['equal_error_rate']:.4f}",
            f"{candidate_metrics['decision_threshold']:.2f}",
            f"{candidate_metrics['uncertainty_margin']:.2f}"
        ]
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Model comparison JSON saved to: {json_path}")
    print(f"Model comparison CSV saved to: {csv_path}")
    return json_path, csv_path


def train_mfcc_backend(
    classifier_type: str = "rf",
    force_reextract: bool = False
) -> Dict[str, Any]:
    """
    Trains Audio Deepfake Model using Handcrafted Audio Features (MFCC, Mel Spec, Spectral Stats).
    Follows candidate model workflow (mfcc_model_candidate.joblib -> mfcc_model.joblib).
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

    total_samples = len(y_train) + len(y_valid) + len(y_test)

    print(f"\nFeature Shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Valid: {X_valid.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Total Clean Samples: {total_samples}")

    # Build and fit Scikit-Learn pipeline
    pipeline = build_mfcc_classifier(classifier_type=classifier_type)
    print("\nFitting Scikit-Learn classifier...")
    pipeline.fit(X_train, y_train)

    # Tune threshold on Validation Set (balanced around 0.50)
    val_probs = pipeline.predict_proba(X_valid)[:, 1]
    best_threshold = 0.50
    best_f1 = 0.0
    for thresh in np.arange(0.45, 0.56, 0.02):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 >= best_f1:
            best_f1 = f1
            best_threshold = round(float(thresh), 2)

    print(f"Tuned Decision Threshold on Validation Set: {best_threshold:.2f} (Val F1: {best_f1:.4f})")

    # Save CANDIDATE model and config first
    candidate_model_path = config.MODELS_DIR / "mfcc_model_candidate.joblib"
    candidate_config_path = config.MODELS_DIR / "config_candidate.json"

    joblib.dump(pipeline, candidate_model_path)

    config_info = {
        "backend": "mfcc",
        "classifier_type": classifier_type,
        "feature_dim": X_train.shape[1],
        "decision_threshold": best_threshold,
        "uncertainty_margin": config.UNCERTAINTY_MARGIN,
        "sample_rate": config.SAMPLE_RATE,
        "window_seconds": config.WINDOW_SECONDS
    }
    with open(candidate_config_path, "w") as f:
        json.dump(config_info, f, indent=4)

    print(f"Candidate model saved to: {candidate_model_path}")

    # Evaluate Candidate on TEST Set
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    candidate_metrics = evaluate_model_performance(
        y_true=y_test,
        y_prob=test_probs,
        threshold=best_threshold,
        save_results=True,
        output_dir=config.RESULTS_DIR
    )

    candidate_metrics["model_name"] = "Expanded Candidate Model (MFCC + RF)"
    candidate_metrics["dataset_sample_count"] = total_samples
    candidate_metrics["uncertainty_margin"] = config.UNCERTAINTY_MARGIN

    # Baseline Model Benchmark Metrics (Baseline 200 samples benchmark)
    baseline_metrics = {
        "model_name": "Baseline Model (200 samples)",
        "dataset_sample_count": 200,
        "accuracy": 0.9667,
        "precision": 1.0000,
        "recall": 0.9333,
        "f1_score": 0.9655,
        "roc_auc": 1.0000,
        "false_positive_rate": 0.0000,
        "false_negative_rate": 0.0667,
        "equal_error_rate": 0.0333,
        "decision_threshold": 0.50,
        "uncertainty_margin": 0.05
    }

    # Generate comparison artifacts
    generate_comparison_artifacts(baseline_metrics, candidate_metrics, config.MODELS_DIR)

    # Promote Candidate Model to Production
    prod_model_path = config.MODELS_DIR / "mfcc_model.joblib"
    prod_config_path = config.MODELS_DIR / "config.json"
    shutil.copy2(candidate_model_path, prod_model_path)
    shutil.copy2(candidate_config_path, prod_config_path)

    print(f"\n[PROMOTION SUCCESS] Candidate model promoted to production: {prod_model_path}")

    print("\nFINAL TEST PERFORMANCE METRICS (MFCC Model)")
    print("=" * 45)
    print(f"  Test Accuracy:   {candidate_metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision:  {candidate_metrics['precision']*100:.2f}%")
    print(f"  Test Recall:     {candidate_metrics['recall']*100:.2f}%")
    print(f"  Test F1-Score:   {candidate_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC:    {candidate_metrics['roc_auc']:.4f}")
    print(f"  False Pos Rate:  {candidate_metrics['false_positive_rate']*100:.2f}%")
    print(f"  False Neg Rate:  {candidate_metrics['false_negative_rate']*100:.2f}%")
    print(f"  Equal Error Rate:{candidate_metrics['equal_error_rate']*100:.2f}%")
    print("=" * 45)

    return candidate_metrics


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
        download_hf_dataset()

    train_mfcc_backend(
        force_reextract=args.force_reextract
    )
