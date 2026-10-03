"""
Training Script for Audio Deepfake Detection Models (Wav2Vec 2.0 & MFCC Fallback)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.metrics import f1_score

import config
from features import Wav2VecExtractor, HandcraftedFeatureExtractor
from model import Wav2VecClassifierHead, build_mfcc_classifier
from evaluate import evaluate_model_performance, calculate_metrics
from download_dataset import download_hf_dataset


def train_wav2vec_backend(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    force_reextract: bool = False
) -> Dict[str, Any]:
    """
    Trains Wav2Vec 2.0 classification head on CPU using cached 768-dim temporal embeddings.
    """
    print("\n" + "=" * 50)
    print("      TRAINING WAV2VEC 2.0 CLASSIFIER (CPU)")
    print("=" * 50)

    # 1. Feature Extraction / Loading
    extractor = Wav2VecExtractor()

    train_cache = config.FEATURES_DIR / "wav2vec_train.pt"
    valid_cache = config.FEATURES_DIR / "wav2vec_valid.pt"
    test_cache = config.FEATURES_DIR / "wav2vec_test.pt"

    X_train, y_train, _ = extractor.cache_split_features(config.TRAIN_DIR, train_cache, force=force_reextract)
    X_valid, y_valid, _ = extractor.cache_split_features(config.VALID_DIR, valid_cache, force=force_reextract)
    X_test, y_test, _ = extractor.cache_split_features(config.TEST_DIR, test_cache, force=force_reextract)

    print(f"\nFeature Shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Valid: {X_valid.shape}")
    print(f"  Test:  {X_test.shape}")

    # 2. Prepare PyTorch DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).float())
    valid_dataset = TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 3. Instantiate Model, Loss, Optimizer
    device = torch.device("cpu")
    head_model = Wav2VecClassifierHead(input_dim=config.WAV2VEC_EMBEDDING_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(head_model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    # 4. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        head_model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = head_model(batch_x).squeeze(1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(X_train)

        # Validation phase
        head_model.eval()
        val_loss = 0.0
        val_probs = []

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                logits = head_model(batch_x).squeeze(1)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.numpy())

        val_loss /= len(X_valid)
        val_probs = np.array(val_probs)

        # Quick validation metrics at threshold 0.5
        val_preds = (val_probs >= 0.50).astype(int)
        val_metrics = calculate_metrics(y_valid, val_preds, val_probs)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | Valid Acc: {val_metrics['accuracy']*100:.2f}% | Valid F1: {val_metrics['f1_score']:.4f}")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = head_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    # 5. Restore Best Model State
    if best_model_state is not None:
        head_model.load_state_dict(best_model_state)

    # 6. Tune Decision Threshold on Validation Dataset
    head_model.eval()
    with torch.no_grad():
        val_logits = head_model(torch.tensor(X_valid)).squeeze(1)
        val_probs = torch.sigmoid(val_logits).numpy()

    best_threshold = 0.50
    best_f1 = 0.0
    for thresh in np.arange(0.30, 0.71, 0.02):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(thresh), 2)

    print(f"\nTuned Decision Threshold on Validation Set: {best_threshold:.2f} (Val F1: {best_f1:.4f})")

    # 7. Save Wav2Vec Classifier & Config
    model_save_path = config.MODELS_DIR / "wav2vec_classifier.pt"
    torch.save(head_model.state_dict(), model_save_path)

    config_info = {
        "backend": "wav2vec",
        "model_name": config.WAV2VEC_MODEL_NAME,
        "input_dim": config.WAV2VEC_EMBEDDING_DIM,
        "decision_threshold": best_threshold,
        "uncertainty_margin": config.UNCERTAINTY_MARGIN,
        "sample_rate": config.SAMPLE_RATE,
        "window_seconds": config.WINDOW_SECONDS,
        "validation_loss": float(best_val_loss)
    }
    with open(config.MODELS_DIR / "config.json", "w") as f:
        json.dump(config_info, f, indent=4)

    print(f"Model saved to: {model_save_path}")
    print(f"Config saved to: {config.MODELS_DIR / 'config.json'}")

    # 8. Final Evaluation on TEST Dataset ONLY
    print("\nEvaluating final model on unseen TEST dataset...")
    with torch.no_grad():
        test_logits = head_model(torch.tensor(X_test)).squeeze(1)
        test_probs = torch.sigmoid(test_logits).numpy()

    test_metrics = evaluate_model_performance(
        y_true=y_test,
        y_prob=test_probs,
        threshold=best_threshold,
        save_results=True,
        output_dir=config.RESULTS_DIR
    )

    print("\nFINAL TEST PERFORMANCE METRICS (Wav2Vec 2.0)")
    print("=" * 45)
    print(f"  Test Accuracy:   {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision:  {test_metrics['precision']*100:.2f}%")
    print(f"  Test Recall:     {test_metrics['recall']*100:.2f}%")
    print(f"  Test F1-Score:   {test_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print("=" * 45)

    return test_metrics


def train_mfcc_backend(
    classifier_type: str = "rf",
    force_reextract: bool = False
) -> Dict[str, Any]:
    """
    Trains CPU Fallback Model using Handcrafted Audio Features (MFCC, Mel Spec, Spectral Stats).
    """
    print("\n" + "=" * 50)
    print("   TRAINING HANDCRAFTED / MFCC FALLBACK CLASSIFIER")
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

    print("\nFINAL TEST PERFORMANCE METRICS (MFCC Fallback)")
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
        default="wav2vec",
        choices=["wav2vec", "mfcc"],
        help="Model backend choice: 'wav2vec' (default) or 'mfcc' (fallback)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs (default 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size (default 4)"
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

    if args.backend.lower() == "wav2vec":
        train_wav2vec_backend(
            epochs=args.epochs,
            batch_size=args.batch_size,
            force_reextract=args.force_reextract
        )
    else:
        train_mfcc_backend(
            force_reextract=args.force_reextract
        )
