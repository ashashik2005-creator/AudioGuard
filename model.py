"""
Machine Learning Models for Audio Deepfake Classification
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def build_mfcc_classifier(classifier_type: str = "rf") -> Pipeline:
    """
    Builds a Scikit-Learn pipeline for Handcrafted/MFCC feature classification.
    classifier_type options: 'rf' (RandomForest), 'svm' (Support Vector Machine), 'lr' (Logistic Regression)
    """
    scaler = StandardScaler()

    if classifier_type.lower() == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight={0: 1.25, 1: 1.0},
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
    elif classifier_type.lower() == "svm":
        clf = SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=config.RANDOM_SEED
        )
    elif classifier_type.lower() == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", clf)
    ])
    return pipeline
