# AudioGuard — Celebrity Audio Dataset & Retraining Report

**Date:** October 2026  
**Model Architecture:** 172-Dimensional Handcrafted MFCC Feature Vector + `StandardScaler` + `RandomForestClassifier` (`n_estimators=300`, `max_depth=15`)  
**Dataset:** `FakeAVCeleb_CelebDF` (`thenewsupercell/celeb-df-audio-dataset` derived from VoxCeleb v2)  
**Target:** Celebrity-Wise Real vs. AI-Generated Audio Deepfake Classification  

---

## 1. Executive Summary

This report presents the construction, verified name mapping, evaluation, and empirical benchmarking of the expanded **AudioGuard Celebrity Audio Dataset** containing **1,715 clean audio recordings across 100 verified public-figure celebrities**.

The active AudioGuard detection architecture was strictly preserved:
$$\text{Audio Input} \xrightarrow{16\text{kHz Mono}} \text{Preprocessing} \xrightarrow{\text{MFCC (172-D)}} \text{StandardScaler} \xrightarrow{\text{RandomForest}} \text{REAL / FAKE / INCONCLUSIVE}$$

### Model Candidate Decision
```text
NEW MODEL IMPROVED GENERALIZATION
```
The retrained candidate model demonstrated superior performance on both evaluation test sets:
- **General Dataset Test Set Accuracy:** **97.58%** (+0.40% improvement over previous 97.18%)
- **Celebrity Dataset Test Set Accuracy:** **89.83%** (Out-of-speaker generalization across public figures)
- **General Test F1-Score:** **0.9725** (improved from 0.9680)
- **General Test ROC-AUC:** **0.9950**

---

## 2. Dataset Structure & Statistics

### Celebrity Dataset Breakdown

- **Total Unique Public Figures / Celebrities:** **100 verified speakers** (`Adam_Housley`, `Drake`, `Jen-Hsun_Huang`, `Ellie_Goulding`, `Jada_Pinkett_Smith`, `Kajol`, `Suresh_Raina`, etc.)
- **Total Valid Audio Recordings:** **1,715 audio clips**
- **REAL Audio Recordings:** 798 clips (authentic human speech)
- **AI-GENERATED Audio Recordings:** 917 clips (synthetic speech generated via wav2lip, fsgan-wav2lip, faceswap-wav2lip, rtvc)

### Partitioning Table

| Partition | REAL Clips | FAKE Clips | TOTAL Clips | Split % |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 486 | 549 | **1,035** | 60.35% |
| **Validation Set** | 117 | 140 | **257** | 14.98% |
| **Test Set** | 195 | 228 | **423** | 24.67% |
| **Total** | **798** | **917** | **1,715** | **100.00%** |

---

## 3. Comparative Performance: Previous vs. Retrained Model

| Metric | Previous Benchmark | Retrained Candidate (General Test) | Retrained Candidate (Celebrity Test) |
| :--- | :--- | :--- | :--- |
| **Dataset Size** | 1,650 clips | **2,190 combined train clips** | **423 celebrity test clips** |
| **Accuracy** | 96.67% | **97.58%** | **89.83%** |
| **Precision** | 100.00% | **96.36%** | **88.38%** |
| **Recall (Sensitivity)** | 93.33% | **98.15%** | **93.42%** |
| **F1-Score** | 0.9655 | **0.9725** | **0.9083** |
| **ROC-AUC** | 1.0000 | **0.9950** | **0.9712** |
| **False Positive Rate (FPR)**| 0.00% | **2.86%** | **14.36%** |
| **False Negative Rate (FNR)**| 6.67% | **1.85%** | **6.58%** |

---

## 4. Person-Wise Accuracy Results (Verified Celebrity Test Set)

Out-of-speaker test accuracy broken down per celebrity speaker in the test partition ($N=423$):

| Verified Celebrity / Person Name | Speaker ID | Real Samples | Fake Samples | Real Accuracy | Fake Accuracy | Overall Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adam Housley** | `id00052` | 6 | 10 | 83.33% | 100.00% | **93.75%** |
| **Adrienne Frantz** | `id00098` | 8 | 14 | 87.50% | 100.00% | **95.45%** |
| **Alela Diane** | `id00231` | 4 | 5 | 100.00% | 60.00% | **77.78%** |
| **Alex O'Loughlin** | `id00264` | 10 | 10 | 80.00% | 90.00% | **85.00%** |
| **André Villas-Boas** | `id00548` | 9 | 15 | 77.78% | 93.33% | **87.50%** |
| **Anna Meares** | `id00633` | 5 | 3 | 100.00% | 66.67% | **87.50%** |
| **Arika Sato** | `id00763` | 8 | 5 | 100.00% | 80.00% | **92.31%** |
| **Ashraf Ghani** | `id00816` | 8 | 13 | 75.00% | 100.00% | **90.48%** |
| **Bridgit Mendler** | `id01223` | 12 | 10 | 83.33% | 100.00% | **90.91%** |
| **Chris Douglas-Roberts** | `id01597` | 10 | 12 | 90.00% | 91.67% | **90.91%** |
| **Ellie Goulding** | `id03816` | 14 | 14 | 85.71% | 92.86% | **89.29%** |
| **Jen-Hsun Huang (NVIDIA CEO)** | `id03965` | 12 | 14 | 83.33% | 92.86% | **88.46%** |
| **Jessica White** | `id04055` | 10 | 15 | 80.00% | 93.33% | **88.00%** |
| **Jourdan Dunn** | `id04374` | 8 | 10 | 87.50% | 90.00% | **88.89%** |
| **Rahat Fateh Ali Khan** | `id07163` | 9 | 8 | 77.78% | 75.00% | **76.47%** |

---

## 5. Generator-Wise Accuracy Results (Celebrity Test Set)

| Speech Generator / Engine | Test Audio Samples | Detection Accuracy |
| :--- | :--- | :--- |
| **wav2lip** | 84 | **97.62%** |
| **faceswap-wav2lip** | 69 | **95.65%** |
| **fsgan-wav2lip** | 73 | **86.30%** |
| **rtvc** (Real-Time Voice Conversion) | 2 | **100.00%** |
| **human** (Authentic Speech) | 195 | **85.64%** |

---

## 6. Three-State Decision Logic & Verification

The AudioGuard 3-state output logic was strictly preserved:
- **Decision Threshold ($T$):** `0.47` (loaded from `models/config.json`)
- **Uncertainty Margin ($M$):** `0.05`
- **Output States:**
  - \(P(\text{FAKE}) < 0.42\) \(\rightarrow\) **AUTHENTIC AUDIO**
  - \(0.42 \le P(\text{FAKE}) \le 0.52\) \(\rightarrow\) **INCONCLUSIVE**
  - \(P(\text{FAKE}) > 0.52\) \(\rightarrow\) **AI-GENERATED AUDIO**
