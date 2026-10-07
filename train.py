"""
AudioGuard — Full Retraining Engine (Combined General & Celebrity Audio Datasets)
Optimized for Generalization & Balanced Real Speech Recognition (Low False-Positive Rate)
"""

import os
import json
import csv
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import config
from features import HandcraftedFeatureExtractor
from model import build_mfcc_classifier
from evaluate import evaluate_model_performance, calculate_metrics
from download_dataset import download_hf_dataset

CELEB_DIR = Path("data/celebrity_audio")
METADATA_CSV = CELEB_DIR / "metadata.csv"


def extract_celeb_split_features(split_name: str, extractor: HandcraftedFeatureExtractor, cache_file: Path, force: bool = False):
    split_dir = CELEB_DIR / split_name
    if cache_file.exists() and not force:
        print(f"Loading cached celebrity {split_name} features from {cache_file}...")
        c_data = joblib.load(cache_file)
        return c_data["X"], c_data["y"], c_data["records"]

    print(f"Extracting 172-D features for celebrity {split_name} split...")
    features = []
    labels = []
    records = []

    if not METADATA_CSV.exists():
        return np.empty((0, 172), dtype=np.float32), np.empty((0,), dtype=np.int64), []

    df_meta = pd.read_csv(METADATA_CSV)
    split_meta = df_meta[df_meta["split"] == split_name]

    for idx, row in split_meta.iterrows():
        p_name = row["person"]
        lbl_str = row["label"]
        fname = row["filename"]
        audio_path = split_dir / p_name / lbl_str / fname

        if not audio_path.exists():
            continue

        try:
            feat = extractor.extract_file_features(audio_path)
            features.append(feat)
            lbl_int = 1 if lbl_str == "fake" else 0
            labels.append(lbl_int)
            records.append({
                "path": str(audio_path),
                "person": p_name,
                "speaker_id": row["speaker_id"],
                "label": lbl_str,
                "label_int": lbl_int,
                "generator": row["generator"]
            })
        except Exception as e:
            print(f"Skipping audio {audio_path}: {e}")

    X = np.array(features, dtype=np.float32) if len(features) > 0 else np.empty((0, 172), dtype=np.float32)
    y = np.array(labels, dtype=np.int64) if len(labels) > 0 else np.empty((0,), dtype=np.int64)

    c_data = {"X": X, "y": y, "records": records}
    joblib.dump(c_data, cache_file)
    print(f"Cached {len(X)} celebrity {split_name} feature vectors to {cache_file}")
    return X, y, records


def train_mfcc_backend(
    classifier_type: str = "rf",
    force_reextract: bool = False
) -> Dict[str, Any]:
    """
    Trains Audio Deepfake Model using Handcrafted Audio Features (MFCC, Mel Spec, Spectral Stats)
    across ALL available training datasets (General Dataset + Celebrity Dataset).
    Calibrates decision threshold for Balanced Accuracy (minimizing false positives on original human speech).
    """
    print("\n" + "=" * 60)
    print("   RETRAINING AUDIOGUARD CLASSIFIER (ALL DATASETS)")
    print("=" * 60)

    extractor = HandcraftedFeatureExtractor()

    # 1. Load General Dataset splits
    gen_train_cache = config.FEATURES_DIR / "mfcc_train.joblib"
    gen_valid_cache = config.FEATURES_DIR / "mfcc_valid.joblib"
    gen_test_cache = config.FEATURES_DIR / "mfcc_test.joblib"

    X_g_train, y_g_train, _ = extractor.cache_split_features(config.TRAIN_DIR, gen_train_cache, force=force_reextract)
    X_g_valid, y_g_valid, _ = extractor.cache_split_features(config.VALID_DIR, gen_valid_cache, force=force_reextract)
    X_g_test, y_g_test, _ = extractor.cache_split_features(config.TEST_DIR, gen_test_cache, force=force_reextract)

    # 2. Load Celebrity Dataset splits (if available)
    celeb_train_cache = config.FEATURES_DIR / "celeb_train.joblib"
    celeb_valid_cache = config.FEATURES_DIR / "celeb_valid.joblib"
    celeb_test_cache = config.FEATURES_DIR / "celeb_test.joblib"

    X_c_train, y_c_train, _ = extract_celeb_split_features("train", extractor, celeb_train_cache, force=force_reextract)
    X_c_valid, y_c_valid, _ = extract_celeb_split_features("valid", extractor, celeb_valid_cache, force=force_reextract)
    X_c_test, y_c_test, _ = extract_celeb_split_features("test", extractor, celeb_test_cache, force=force_reextract)

    # 3. Combine Training Pools
    if len(X_c_train) > 0:
        X_train_full = np.vstack([X_g_train, X_c_train])
        y_train_full = np.hstack([y_g_train, y_c_train])
        X_valid_full = np.vstack([X_g_valid, X_c_valid])
        y_valid_full = np.hstack([y_g_valid, y_c_valid])
    else:
        X_train_full, y_train_full = X_g_train, y_g_train
        X_valid_full, y_valid_full = X_g_valid, y_g_valid

    real_train_cnt = np.sum(y_train_full == 0)
    fake_train_cnt = np.sum(y_train_full == 1)

    print(f"\nUnified Training Dataset Statistics:")
    print(f"  Total Training Samples: {len(X_train_full)}")
    print(f"    - Authentic Real Speech (0):  {real_train_cnt} ({real_train_cnt/len(y_train_full)*100:.1f}%)")
    print(f"    - Synthetic AI Speech  (1):  {fake_train_cnt} ({fake_train_cnt/len(y_train_full)*100:.1f}%)")
    print(f"  Feature Dimension:            {X_train_full.shape[1]}")

    # 4. Build and fit Scikit-Learn Random Forest Pipeline
    pipeline = build_mfcc_classifier(classifier_type=classifier_type)
    print("\nFitting Scikit-Learn Random Forest Classifier...")
    pipeline.fit(X_train_full, y_train_full)

    # 5. Tune Threshold on Validation Set for Balanced Accuracy (Real Acc + Fake Acc)/2
    val_probs = pipeline.predict_proba(X_valid_full)[:, 1]
    best_threshold = 0.50
    best_bal_acc = 0.0

    for thresh in np.arange(0.46, 0.56, 0.01):
        preds = (val_probs >= thresh).astype(int)
        real_acc = accuracy_score(y_valid_full[y_valid_full == 0], preds[y_valid_full == 0]) if np.sum(y_valid_full == 0) > 0 else 0.0
        fake_acc = accuracy_score(y_valid_full[y_valid_full == 1], preds[y_valid_full == 1]) if np.sum(y_valid_full == 1) > 0 else 0.0
        bal_acc = 0.5 * (real_acc + fake_acc)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = round(float(thresh), 2)

    print(f"\nCalibrated Decision Threshold (Balanced Accuracy): {best_threshold:.2f} (Val Balanced Acc: {best_bal_acc*100:.2f}%)")

    # 6. Save Model Artifacts
    candidate_model_path = config.MODELS_DIR / "mfcc_model_candidate.joblib"
    candidate_config_path = config.MODELS_DIR / "config_candidate.json"
    prod_model_path = config.MODELS_DIR / "mfcc_model.joblib"
    prod_config_path = config.MODELS_DIR / "config.json"

    joblib.dump(pipeline, candidate_model_path)

    config_info = {
        "backend": "mfcc",
        "classifier_type": classifier_type,
        "feature_dim": X_train_full.shape[1],
        "decision_threshold": best_threshold,
        "uncertainty_margin": config.UNCERTAINTY_MARGIN,
        "sample_rate": config.SAMPLE_RATE,
        "window_seconds": config.WINDOW_SECONDS
    }
    with open(candidate_config_path, "w") as f:
        json.dump(config_info, f, indent=4)

    # 7. Evaluate on General Test Set
    g_probs = pipeline.predict_proba(X_g_test)[:, 1]
    g_metrics = evaluate_model_performance(
        y_true=y_g_test,
        y_prob=g_probs,
        threshold=best_threshold,
        save_results=True,
        output_dir=config.RESULTS_DIR
    )

    # Promote to Production
    shutil.copy2(candidate_model_path, prod_model_path)
    shutil.copy2(candidate_config_path, prod_config_path)

    print(f"\n[PROMOTION SUCCESS] Model trained across all datasets promoted to production: {prod_model_path}")
    print("\nFINAL GENERAL TEST PERFORMANCE METRICS")
    print("=" * 45)
    print(f"  Test Accuracy:   {g_metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision:  {g_metrics['precision']*100:.2f}%")
    print(f"  Test Recall:     {g_metrics['recall']*100:.2f}%")
    print(f"  Test F1-Score:   {g_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC:    {g_metrics['roc_auc']:.4f}")
    print(f"  False Pos Rate:  {g_metrics['false_positive_rate']*100:.2f}% (False alarm on Real Voice)")
    print(f"  False Neg Rate:  {g_metrics['false_negative_rate']*100:.2f}% (Missed AI Voice)")
    print("=" * 45)

    return g_metrics


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

    # Ensure dataset exists
    if not (config.TRAIN_DIR / "real").exists() or len(list((config.TRAIN_DIR / "real").glob("*.*"))) == 0:
        print("Dataset splits missing in data/processed. Downloading default dataset...")
        download_hf_dataset()

    train_mfcc_backend(
        force_reextract=args.force_reextract
    )
