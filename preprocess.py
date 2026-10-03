"""
Dataset Validation, Deduplication, and Stratified 70/15/15 Train/Validation/Test Split
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
from sklearn.model_selection import train_test_split

import config
from audio_utils import compute_file_hash, load_and_preprocess_audio


def validate_and_deduplicate_raw_dataset(
    raw_dir: Union[str, Path] = config.RAW_DIR,
    max_per_class: Optional[int] = None
) -> Dict[str, Any]:
    """
    Scans data/raw/real and data/raw/fake directories.
    Detects corrupted files, checks exact duplicate files using SHA-256 hashing,
    and returns a structured list of clean, valid files per class.
    """
    raw_dir = Path(raw_dir)
    real_dir = raw_dir / "real"
    fake_dir = raw_dir / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(f"Raw dataset directories missing. Expected {real_dir} and {fake_dir}")

    clean_records = []
    seen_hashes = {}
    corrupted_files = []
    duplicate_files = []

    total_duration_real = 0.0
    total_duration_fake = 0.0

    print("Starting raw dataset validation and SHA-256 deduplication...")

    for label_str, class_dir in [("real", real_dir), ("fake", fake_dir)]:
        if not class_dir.exists():
            continue

        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_FORMATS
        ]
        files.sort()  # Sort for deterministic processing order

        class_clean_count = 0

        for file_path in files:
            # Check max per class limit if specified
            if max_per_class is not None and class_clean_count >= max_per_class:
                break

            # 1. Deduplication check via SHA-256 hash
            try:
                file_hash = compute_file_hash(file_path)
            except Exception as e:
                corrupted_files.append({"path": str(file_path), "reason": f"Hash computation failed: {e}"})
                continue

            if file_hash in seen_hashes:
                original_path = seen_hashes[file_hash]
                duplicate_files.append({
                    "duplicate_path": str(file_path),
                    "original_path": original_path
                })
                continue

            # 2. Audio integrity & loading check
            try:
                waveform, sr, duration = load_and_preprocess_audio(file_path)
                if label_str == "real":
                    total_duration_real += duration
                else:
                    total_duration_fake += duration
            except Exception as e:
                corrupted_files.append({"path": str(file_path), "reason": str(e)})
                continue

            # Mark hash as seen
            seen_hashes[file_hash] = str(file_path)

            clean_records.append({
                "path": file_path,
                "label": label_str,
                "label_int": config.LABEL_TO_INT[label_str],
                "hash": file_hash,
                "duration": round(duration, 2)
            })

            class_clean_count += 1

    report = {
        "total_records": len(clean_records),
        "real_count": sum(1 for r in clean_records if r["label"] == "real"),
        "fake_count": sum(1 for r in clean_records if r["label"] == "fake"),
        "corrupted_count": len(corrupted_files),
        "duplicate_count": len(duplicate_files),
        "corrupted_files": corrupted_files,
        "duplicate_files": duplicate_files,
        "records": clean_records
    }

    print("\nDataset Validation Report")
    print("-" * 30)
    print(f"Total Valid Files:    {report['total_records']}")
    print(f"  Real Samples:       {report['real_count']}")
    print(f"  Fake Samples:       {report['fake_count']}")
    print(f"Corrupted Files:      {report['corrupted_count']}")
    print(f"Duplicate Files:      {report['duplicate_count']}")
    print("-" * 30)

    return report


def create_stratified_splits(
    raw_dir: Union[str, Path] = config.RAW_DIR,
    processed_dir: Union[str, Path] = config.PROCESSED_DIR,
    max_per_class: Optional[int] = None,
    seed: int = config.RANDOM_SEED
) -> Dict[str, Any]:
    """
    Performs a leak-free, deduplicated, stratified 70% TRAIN / 15% VALID / 15% TEST split.
    Copies audio files to data/processed/{train,valid,test}/{real,fake}.
    """
    validation_report = validate_and_deduplicate_raw_dataset(raw_dir, max_per_class=max_per_class)
    records = validation_report["records"]

    if len(records) < 10:
        raise ValueError(f"Too few valid audio records ({len(records)}) to split dataset.")

    file_paths = [r["path"] for r in records]
    labels = [r["label_int"] for r in records]

    # First split: 70% Train, 30% Temp (Valid + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=seed
    )

    # Second split: Split 30% Temp equally into 15% Valid and 15% Test
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=seed
    )

    splits = {
        "train": (train_paths, train_labels),
        "valid": (valid_paths, valid_labels),
        "test": (test_paths, test_labels)
    }

    processed_dir = Path(processed_dir)

    # Clear existing processed directories
    for split_name in ["train", "valid", "test"]:
        for cls in ["real", "fake"]:
            out_folder = processed_dir / split_name / cls
            if out_folder.exists():
                shutil.rmtree(out_folder)
            out_folder.mkdir(parents=True, exist_ok=True)

    manifest_stats = {"total_files": len(records), "splits": {}}

    print("\nCopying processed audio files to dataset splits...")
    for split_name, (paths, lbls) in splits.items():
        real_c = 0
        fake_c = 0

        for src_path, lbl in zip(paths, lbls):
            cls_str = config.INT_TO_LABEL[lbl].lower()
            dst_folder = processed_dir / split_name / cls_str
            dst_path = dst_folder / Path(src_path).name

            shutil.copy2(src_path, dst_path)

            if cls_str == "real":
                real_c += 1
            else:
                fake_c += 1

        manifest_stats["splits"][split_name] = {
            "total": len(paths),
            "real": real_c,
            "fake": fake_c
        }

    # Save manifest JSON
    manifest_path = processed_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "validation_summary": {
                "total_valid": len(records),
                "real_count": validation_report["real_count"],
                "fake_count": validation_report["fake_count"],
                "corrupted_count": validation_report["corrupted_count"],
                "duplicate_count": validation_report["duplicate_count"]
            },
            "splits": manifest_stats["splits"]
        }, f, indent=4)

    print("\nDataset Split Summary")
    print("=" * 40)
    for s_name, s_info in manifest_stats["splits"].items():
        print(f"{s_name.upper():<10} -> Total: {s_info['total']:<4} | Real: {s_info['real']:<4} | Fake: {s_info['fake']}")
    print("=" * 40)
    print(f"Manifest saved to: {manifest_path}\n")

    return manifest_stats
