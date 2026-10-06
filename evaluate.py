"""
Model Evaluation and Metrics Reporting (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import config


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, TP, TN, FP, FN.
    FAKE (1) is the positive class.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.50

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    try:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
        fnr_curve = 1.0 - tpr_curve
        eer_idx = np.nanargmin(np.abs(fpr_curve - fnr_curve))
        eer = float((fpr_curve[eer_idx] + fnr_curve[eer_idx]) / 2.0)
    except Exception:
        eer = 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "equal_error_rate": float(eer),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: Union[str, Path] = config.RESULTS_DIR / "confusion_matrix.png"):
    """
    Renders and saves Confusion Matrix image.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix (Positive Class = FAKE)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("Actual True Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Union[str, Path] = config.RESULTS_DIR / "roc_curve.png"):
    """
    Renders and saves Receiver Operating Characteristic (ROC) Curve image.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.50

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_model_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = config.DEFAULT_DECISION_THRESHOLD,
    save_results: bool = True,
    output_dir: Union[str, Path] = config.RESULTS_DIR
) -> Dict[str, Any]:
    """
    Evaluates model probabilities against decision threshold and generates plots & metrics.json.
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics["decision_threshold"] = float(threshold)

    if save_results:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plot_confusion_matrix(cm, save_path=output_dir / "confusion_matrix.png")
        plot_roc_curve(y_true, y_prob, save_path=output_dir / "roc_curve.png")

        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics
