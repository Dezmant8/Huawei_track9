# Report: Robustness of VLMs for Table Parsing under Image Rotation

## 1. Introduction

### 1.1 Problem Statement

The goal of this work is to evaluate and improve the robustness of Vision-Language Models (VLMs) to image rotation in table parsing. VLMs are trained on horizontally oriented images, and their performance can degrade significantly when the input image is rotated by 90°, 180°, or 270°.

### 1.2 Motivation

In real-world scenarios (document scanning, photographing), table images are often rotated. This is critical for:
- Automated processing of financial documents
- Archive digitization
- OCR pipelines in mobile applications

### 1.3 Approach

1. Built a benchmark of 1000 rotated tables (SynthTabNet)
2. Evaluated 2 VLM models as baselines (without rotation correction)
3. Trained a rotation classifier (ResNet-18) for orientation correction
4. Built the final pipeline: classifier → correction → VLM

---

## 2. Data

### 2.1 SynthTabNet

A synthetic table dataset (IBM Research), 600k images, 4 subsets:

| Subset | Style | Characteristics |
|---|---|---|
| fintabnet | Financial tables | Complex merged cells, numerical data |
| marketing | Marketing tables | Colorful design, colored backgrounds |
| pubtabnet | Scientific publications | Strict style, text-heavy |
| sparse | Sparse tables | Few cells, lots of whitespace |

### 2.2 Benchmark

- 250 test samples × 4 parts = 1000 images
- Deterministic rotation: `index % 4 → {0°, 90°, 180°, 270°}`
- 250 images per rotation angle
- Ground truth: HTML tables from JSON annotations (format_html + normalize_html_for_teds)

### 2.3 Benchmark Statistics

- Average image size: 481 × 355 px (min 206×74, max 512×512)
- Average complexity: 95.1 cells, 13.4 rows, 6.7 columns, 2.9 merged cells
- Max complexity: 232 cells, 25 rows
- All 1000 samples passed validation (0 errors, 0 empty GT HTML)

---

## 3. Methodology

### 3.1 Baseline Models

#### TRivia-3B (Expert VLM)
- **Architecture**: Qwen2.5-VL-3B + self-supervised fine-tuning on tables
- **Output format**: OTSL (Object Table Structure Language)
- **Conversion**: OTSL → HTML via `convert_otsl_to_html()`
- **Rationale**: state-of-the-art on OmniDocBench, specialized for tables

#### Qwen2.5-VL-3B-Instruct (General VLM)
- **Architecture**: Qwen2.5-VL-3B (general-purpose VLM)
- **Output format**: direct HTML via prompt
- **Rationale**: baseline for comparison — shows how much table specialization (TRivia) outperforms a general model

### 3.2 Metrics

#### TEDS (Tree Edit Distance based Similarity)
- Converts HTML tables to trees
- Computes tree edit distance (APTED)
- TEDS = 1 - distance / max_nodes
- Range: [0, 1], 1 = identical tables

#### TEDS-struct
- Same as TEDS, but ignores cell content
- Evaluates structure only: number of rows/columns, rowspan/colspan

### 3.3 Rotation Classifier

- **Architecture**: ResNet-18 (ImageNet pretrained), FC → Linear(512, 4)
- **Data**: 20000 images × 4 parts × 4 rotations = 80000 train, 8000 val
- **Training**: Adam, lr=1e-3, cosine annealing, early stopping (patience=3)
- **Gradient clipping**: max_norm=1.0 for training stability during fine-tuning
- **Augmentations**: RandomResizedCrop, ColorJitter (no HorizontalFlip — it would break orientation labels)

#### 3.3.1 Rationale for ResNet-18

| Criterion | ResNet-18 | ResNet-34 | EfficientNet-B0 |
|---|---|---|---|
| Parameters | 11.7M | 21.8M | 5.3M |
| Training speed | Fast (~50 min) | Medium | Medium |
| Expected accuracy | >98% | >98% | >97% |
| Transfer learning stability | High | High | Medium |

ResNet-18 was chosen as the optimal trade-off: powerful enough for a 4-class task, yet fast and stable during fine-tuning.

### 3.4 Final Pipeline

```
Rotated image
    → ResNet-18 classifier → predicted angle + confidence
    → Unload classifier from GPU (free MPS memory)
    → VLM inference (TRivia-3B / Qwen2.5-VL-3B)
    → HTML prediction
    → TEDS evaluation
```

An important engineering detail: the classifier and VLM run **sequentially** (not simultaneously in memory). First, all rotation angles are predicted (phase 1, ~1 sec for 200 samples), then the classifier is unloaded from MPS, and only then the VLM is loaded (phase 2). This solves the MPS memory pressure issue on Apple Silicon, which caused VLM generation degradation.

---

## 4. Results

> **Note**: experiments were conducted on a subset of 200 samples (fintabnet, 50 per angle). This provides statistically significant results within a reasonable inference time.

### 4.1 Baseline (without rotation correction)

| Model | TEDS | ±std | TEDS-struct | @0° | @90° | @180° | @270° |
|---|---|---|---|---|---|---|---|
| TRivia-3B | 0.6010 | 0.3258 | 0.7326 | 0.9682 | 0.4436 | 0.3460 | 0.6464 |
| Qwen2.5-VL-3B | 0.3435 | 0.3173 | 0.5307 | 0.5951 | 0.2750 | 0.1994 | 0.3044 |

**Key observation**: both models show sharp degradation on rotated images. TRivia at 0° achieves TEDS=0.97, but at 180° — only 0.35 (a 64% drop). Qwen is even worse: 0.60 at 0° and 0.20 at 180°.

### 4.2 Rotation Classifier

- **Architecture**: ResNet-18 (11,178,564 parameters)
- **Best validation accuracy**: **99.69%** (7975/8000)
- **Trivial baseline** (always 0°): 25.0%
- **Improvement over baseline**: +74.69%
- **Training**: 10 epochs, ~50 min on Apple MPS (M1/M2/M3)

#### Confusion Matrix (val set, 8000 samples)

| True \ Pred | 0° | 90° | 180° | 270° |
|---|---|---|---|---|
| **0°** | 1996 | 0 | 3 | 1 |
| **90°** | 1 | 1994 | 0 | 5 |
| **180°** | 11 | 0 | 1989 | 0 |
| **270°** | 0 | 4 | 0 | 1996 |

#### Per-class Metrics

| Angle | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0° | 0.9940 | 0.9980 | 0.9960 | 2000 |
| 90° | 0.9980 | 0.9970 | 0.9975 | 2000 |
| 180° | 0.9985 | 0.9945 | 0.9965 | 2000 |
| 270° | 0.9970 | 0.9980 | 0.9975 | 2000 |

On the test subset of 200 benchmark samples, the classifier achieved **100% accuracy** (200/200).

#### Training Dynamics

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|---|---|---|---|---|---|
| 1 | 0.0534 | 0.9817 | 0.1106 | 0.9603 | 9.76e-4 |
| 2 | 0.0167 | 0.9949 | 0.0533 | 0.9798 | 9.05e-4 |
| 5 | 0.0058 | 0.9983 | 0.0262 | 0.9914 | 5.00e-4 |
| 7 | 0.0017 | 0.9994 | 0.0155 | 0.9949 | 2.06e-4 |
| 10 | 0.0006 | 0.9998 | 0.0110 | 0.9969 | 0.00e+0 |

### 4.3 Pipeline (with rotation correction)

| Model | TEDS | ±std | TEDS-struct | @0° | @90° | @180° | @270° |
|---|---|---|---|---|---|---|---|
| TRivia-3B + RotCls | **0.9531** | 0.0701 | **0.9633** | 0.9682 | 0.9398 | 0.9459 | 0.9587 |
| Qwen2.5-VL-3B + RotCls | **0.6181** | 0.2762 | **0.7351** | 0.5951 | 0.6006 | 0.6621 | 0.6147 |

### 4.4 Comparison: Baseline vs Pipeline

| Model | Baseline TEDS | Pipeline TEDS | ΔTEDS | Baseline TEDS-s | Pipeline TEDS-s | ΔTEDS-s |
|---|---|---|---|---|---|---|
| TRivia-3B | 0.6010 | **0.9531** | **+0.3521** | 0.7326 | **0.9633** | **+0.2307** |
| Qwen2.5-VL-3B | 0.3435 | **0.6181** | **+0.2747** | 0.5307 | **0.7351** | **+0.2044** |

#### Bootstrap 95% CI

| Model | Baseline CI | Pipeline CI |
|---|---|---|
| TRivia-3B | [0.5526 — 0.6457] | [0.9435 — 0.9621] |
| Qwen2.5-VL-3B | [0.3011 — 0.3861] | [0.5799 — 0.6575] |

The confidence intervals do not overlap — the improvement is statistically significant.

---

## 5. Analysis

### 5.1 Impact of Rotation on Baselines

VLMs are **highly sensitive** to image orientation:

- **TRivia-3B**: TEDS drops from 0.97 (0°) to 0.35 (180°) — a **64% loss** in quality. Interestingly, 270° (TEDS=0.65) is significantly better than 90° (TEDS=0.44), which may be related to attention pattern characteristics in the Qwen2.5-VL ViT encoder.
- **Qwen2.5-VL**: even more pronounced degradation — TEDS drops from 0.60 to 0.20 at 180°. The general VLM handles tables worse even at correct orientation (0.60 vs 0.97 for TRivia).

180° is the worst angle for both models. This is explained by the fact that at 180° the text is completely upside-down and unreadable, whereas at 90°/270° some structural cues (vertical/horizontal lines) can still be partially interpreted.

### 5.2 Effectiveness of Rotation Correction

The pipeline with rotation correction **dramatically improves** results:

- **TRivia-3B**: TEDS increased from 0.60 to 0.95 (+0.35). All angles show consistently high quality (0.94–0.97), std decreased from 0.33 to 0.07 — the model became **significantly more stable**.
- **Qwen2.5-VL**: TEDS increased from 0.34 to 0.62 (+0.27). Quality evened out across angles (0.60–0.66), but remains below TRivia due to the lack of table specialization.

Notably, the TRivia pipeline on rotated images (90°: 0.94, 180°: 0.95) achieves results close to the baseline at 0° (0.97). This confirms that the classifier effectively solves the orientation problem.

### 5.3 Error Analysis

#### TRivia-3B Baseline
- 67.0% acceptable (TEDS ≥ 0.5)
- 16.5% poor structure (TEDS-struct < 0.5)
- 16.5% poor content (structure OK, content wrong)
- 0% empty predictions

#### TRivia-3B Pipeline
- **99.5% acceptable** (TEDS ≥ 0.5)
- 0% poor structure
- 0.5% poor content (1 sample)
- 0% empty predictions

#### Qwen2.5-VL Baseline
- 40.0% acceptable
- 44.5% poor structure — Qwen often misidentifies rowspan/colspan
- 15.5% poor content
- 0% empty predictions

#### Qwen2.5-VL Pipeline
- **78.5% acceptable** (+38.5%)
- 16.0% poor structure
- 5.5% poor content
- 0% empty predictions

Worst-case analysis for TRivia pipeline: the single difficult sample — idx=2 (fintabnet, 180°, TEDS=0.49) — is a large financial table with 180+ cells, where even after correction the model misidentifies several merged cells.

### 5.4 Statistical Significance

Wilcoxon signed-rank test (paired non-parametric test):

| Comparison | p-value | Significant | Cohen's d | Effect size |
|---|---|---|---|---|
| TRivia baseline → pipeline | 1.26 × 10⁻²⁵ | *** (p < 0.001) | 1.494 | Very large |
| Qwen baseline → pipeline | 3.52 × 10⁻²³ | *** (p < 0.001) | 0.923 | Large |

Both improvements are **highly significant** (p ≪ 0.001). Cohen's d > 0.8 for both — **large effect size**. For TRivia d=1.49 — **very large effect**, confirming the practical significance of rotation correction.

### 5.5 TRivia vs Qwen Comparison

| Aspect | TRivia-3B | Qwen2.5-VL-3B |
|---|---|---|
| TEDS at 0° (baseline) | 0.968 | 0.595 |
| TEDS pipeline (overall) | **0.953** | 0.618 |
| Output format | OTSL (compact) | HTML (verbose) |
| Inference (sec/img) | ~24 | ~79 |
| Specialization | Tables | General VLM |

TRivia significantly outperforms Qwen on table recognition: +0.34 TEDS in pipeline. This confirms the value of specialized fine-tuning for tables. Additionally, TRivia is 3.3× faster thanks to the compact OTSL format.

---

## 6. Conclusion

### Key Findings:

1. **VLMs are highly sensitive to rotation**: TRivia loses up to 64% of quality (TEDS: 0.97→0.35) when rotated by 180°. Qwen is even more vulnerable (0.60→0.20). This is a critical limitation for real-world applications.

2. **The rotation classifier solves the problem**: ResNet-18 with ImageNet pretrained weights achieves 99.69% accuracy on the 4-class orientation task. On benchmark test data — 100%. Training takes ~50 min on Apple MPS.

3. **The pipeline restores quality**: TRivia + classifier achieves TEDS=0.953 on rotated images — practically at the level of baseline at 0° (0.968). The improvement is highly statistically significant (p=1.26×10⁻²⁵, Cohen's d=1.49).

4. **Table specialization is critical**: TRivia (expert VLM) outperforms Qwen (general VLM) by +0.34 TEDS in the pipeline, confirming the value of domain-specific fine-tuning.

### Directions for Future Work:

- **Rotation-aware fine-tuning**: training VLMs with rotation augmentation may eliminate the need for a separate classifier
- **Larger VLMs**: 7B+ parameter models (TRivia-7B, Qwen2.5-VL-7B) may be more robust to rotation
- **Arbitrary angles**: extending the classifier to continuous angles (regression instead of 4-class classification) for realistic scanning scenarios
- **Real-world datasets**: validation on real (non-synthetic) tables — PubTabNet, FinTabNet original, ICDAR
- **End-to-end pipeline**: integrating table detection + orientation + recognition into a single pipeline

---

## Appendix: Technical Environment

- **Hardware**: Apple Silicon MacBook (48 GB RAM, MPS)
- **PyTorch**: with MPS support (Metal Performance Shaders)
- **Models**: TRivia-3B (~3 GB), Qwen2.5-VL-3B (~3 GB), ResNet-18 (~45 MB)
- **Inference time**:
  - TRivia baseline: ~12 sec/img
  - TRivia pipeline: ~24 sec/img
  - Qwen baseline: ~163 sec/img
  - Qwen pipeline: ~79 sec/img
- **Classifier training time**: ~50 min (10 epochs, 80k samples)
- **Seed**: 42 (fixed for random, numpy, torch)
