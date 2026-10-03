"""
Automatic Downloader for Hugging Face Deepfake Audio Dataset (garystafford/deepfake-audio-detection)
Downloads dataset, validates real/fake labels, deduplicates, and creates stratified train/val/test splits.
"""

import os
import io
import shutil
import argparse
from pathlib import Path
from datasets import load_dataset, Audio

import config
from preprocess import create_stratified_splits


def download_hf_dataset(max_per_class: int = None, force: bool = False):
    """
    Downloads garystafford/deepfake-audio-detection from Hugging Face,
    decodes audio into data/raw/real and data/raw/fake, and creates splits.
    """
    raw_real_dir = config.RAW_REAL_DIR
    raw_fake_dir = config.RAW_FAKE_DIR

    if force:
        print("Cleaning existing raw dataset files for fresh download...")
        for p in [raw_real_dir, raw_fake_dir]:
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    # Count existing raw files
    existing_real = len(list(raw_real_dir.glob("*.*")))
    existing_fake = len(list(raw_fake_dir.glob("*.*")))

    if existing_real > 0 and existing_fake > 0 and not force:
        print(f"Found existing raw dataset: {existing_real} real, {existing_fake} fake samples.")
        print("Skipping re-download. Use --force-redownload to download again.")
    else:
        print(f"Downloading dataset '{config.HF_DATASET_NAME}' from Hugging Face Hub...")
        print("Dataset info: ~1,866 audio samples (933 REAL, 933 FAKE), 16 kHz FLAC audio.")

        # Load HuggingFace dataset
        ds = load_dataset(config.HF_DATASET_NAME, split="train")
        ds = ds.cast_column("audio", Audio(decode=False))

        real_saved = 0
        fake_saved = 0

        target_per_class = max_per_class if (max_per_class is not None and max_per_class > 0) else 999999

        for sample in ds:
            label_int = sample["label"]  # 0: real, 1: fake
            audio_bytes = sample["audio"]["bytes"]

            if label_int == 0:
                if real_saved >= target_per_class:
                    continue
                file_name = f"real_{real_saved:04d}.flac"
                out_path = raw_real_dir / file_name
                real_saved += 1
            else:
                if fake_saved >= target_per_class:
                    continue
                file_name = f"fake_{fake_saved:04d}.flac"
                out_path = raw_fake_dir / file_name
                fake_saved += 1

            # Save raw FLAC bytes
            with open(out_path, "wb") as f:
                f.write(audio_bytes)

            if real_saved >= target_per_class and fake_saved >= target_per_class:
                break

        print(f"\nSuccessfully saved raw audio files to disk:")
        print(f"  REAL: {real_saved} files saved to {raw_real_dir}")
        print(f"  FAKE: {fake_saved} files saved to {raw_fake_dir}")

    # Run validation, deduplication, and stratified 70/15/15 split
    create_stratified_splits(
        raw_dir=config.RAW_DIR,
        processed_dir=config.PROCESSED_DIR,
        max_per_class=max_per_class,
        seed=config.RANDOM_SEED
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Deepfake Audio Dataset")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum samples per class (e.g. 100, 250, 500, 900)"
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force redownloading from Hugging Face Hub"
    )
    args = parser.parse_args()

    download_hf_dataset(max_per_class=args.max_per_class, force=args.force_redownload)
