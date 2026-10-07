# AudioGuard — Celebrity Audio Dataset Sources & Metadata Documentation

This document records the official dataset source, sample breakdown, metadata schema, audio standardization, and leak-free split partitioning for the AudioGuard celebrity-wise real vs. AI-generated audio dataset.

---

## 1. Primary Dataset Information

- **Dataset Name:** `thenewsupercell/celeb-df-audio-dataset` (derived from FakeAVCeleb & VoxCeleb v2)
- **Official Source Repository:** Hugging Face Hub (`https://huggingface.co/datasets/thenewsupercell/celeb-df-audio-dataset`)
- **Domain:** Celebrity/Public-Figure Audio Deepfake Detection & Synthetic Speech Identification
- **Number of Unique Celebrity Speakers:** **100 public figure speakers** (`id00052`, `id00068`, `id00076`, `id00098`, `id00100`, `id00145`, `id00185`, `id00264`, `id00548`, etc.)
- **Audio Specification:** 16,000 Hz, Mono, Float32, Peak Normalized
- **License / Access:** Public research dataset for non-commercial audio forensic evaluation

---

## 2. Sample Breakdown & Statistics

### Raw Download & Filtering
- **Total Raw Records Ingested:** 4,288 audio clips across 100 celebrity speakers.
- **SHA-256 Deduplication:** Cryptographic hash inspection filtered out duplicate files.
- **Total Clean Audio Files Saved:** **1,715 unique audio files** (798 Real, 917 Fake).

### Partitioning Breakdown (Train / Validation / Test)

| Partition | REAL Clips | FAKE Clips | TOTAL Clips | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 486 | 549 | **1,035** | 60.35% |
| **Validation Set** | 117 | 140 | **257** | 14.98% |
| **Test Set** | 195 | 228 | **423** | 24.67% |
| **Total** | **798** | **917** | **1,715** | **100.00%** |

---

## 3. Metadata Schema (`data/celebrity_audio/metadata.csv`)

The dataset metadata is stored in `data/celebrity_audio/metadata.csv` with the following columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `person` | String | Celebrity / Speaker directory identifier (e.g. `Speaker_id00591`) |
| `speaker_id` | String | Official VoxCeleb / FakeAVCeleb speaker ID (e.g. `id00591`) |
| `label` | String | Audio classification label (`real` or `fake`) |
| `filename` | String | Standardized file name (e.g. `Speaker_id00591_real_001.wav`) |
| `source_dataset` | String | Primary benchmark source (`FakeAVCeleb_CelebDF`) |
| `generator` | String | AI generation / lip-sync engine (`human`, `wav2lip`, `fsgan-wav2lip`, `faceswap-wav2lip`, `rtvc`) |
| `split` | String | Dataset partition (`train`, `valid`, or `test`) |
| `duration` | Float | Audio duration in seconds (4.0s average) |
| `sample_rate` | Integer | Standardized sample rate (`16000` Hz) |

---

## 4. Leak-Free Partitioning & Audio Standardization Protocol

1. **Speaker Isolation & Split Partitioning:** Speakers were partitioned across train, validation, and test sets using fixed random seeds. No single audio recording exists in more than one partition.
2. **Audio Preprocessing:** All audio files were processed using AudioGuard's native pipeline:
   - Resampled to **16,000 Hz Mono**
   - 32-bit Float32 representation
   - Peak amplitude normalized to \([-1.0, +1.0]\)
3. **Exact SHA-256 Deduplication:** Every audio stream was hashed prior to disk export to prevent exact byte duplicates.
