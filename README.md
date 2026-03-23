# Multimodal Document Understanding: Robustness of VLMs for Table Parsing under Rotation

A study of the robustness of Vision-Language Models (VLMs) in recognizing tables under arbitrary image rotation. The project includes benchmark construction, evaluation of baseline VLMs, training a rotation classifier, and building the final orientation-correction pipeline.

## Project Structure

```
.
├── scripts/
│   ├── config.py                      # All hyperparameters and paths
│   ├── utils/
│   │   ├── seed.py                    # Reproducibility (seed, device)
│   │   ├── html_utils.py              # HTML normalization + format_html
│   │   ├── teds.py                    # TEDS / TEDS-struct metric
│   │   └── rotation.py               # Image rotation / correction
│   ├── 01_download_dataset.py         # Step 1: Download SynthTabNet
│   ├── 02_build_benchmark.py          # Step 2: Build benchmark
│   ├── 03_evaluate_baselines.py       # Step 3: Evaluate VLMs without correction
│   ├── 04_train_rotation_classifier.py # Step 4: Train classifier
│   ├── 05_run_pipeline.py             # Step 5: Pipeline (classifier + VLM)
│   └── 06_compute_metrics.py          # Steps 6-7: Metrics and comparison
├── data/
│   ├── raw/                           # Original SynthTabNet (after step 1)
│   └── benchmark/                     # Benchmark (after step 2)
├── models/
│   └── rotation_classifier/           # Classifier weights (after step 4)
├── results/                           # Inference results and metrics
├── SynthTabNet-main/                  # SynthTabNet repository (reference)
├── TRivia-main/                       # TRivia repository (otsl_utils)
├── requirements.txt
├── report.md                          # Report template (fill in with results)
└── README.md                          # This file
```

## Requirements

- Python 3.10+
- ~15 GB of free disk space (SynthTabNet dataset ~12 GB + models ~3 GB)
- GPU recommended (Apple MPS / NVIDIA CUDA), but CPU also works (slowly)
- ~8 GB VRAM for inference with 3B models

## Installation

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note**: `torch` will be installed for CPU by default. For Apple MPS, this works
> out of the box (MPS is supported in PyTorch >= 2.1). For CUDA, see
> [pytorch.org](https://pytorch.org/get-started/locally/).

## Running the Pipeline: 7 Steps

Run all commands from the project root with the venv activated:

```bash
source venv/bin/activate
```

---

### Step 1. Download the SynthTabNet dataset

```bash
python scripts/01_download_dataset.py
```

Downloads 4 SynthTabNet v2.0.0 ZIP archives (~12 GB total) and extracts them into `data/raw/`.
Each archive contains table images and annotations in `synthetic_data.jsonl`.

The script supports resume — if part of the data has already been downloaded, it will be skipped.
To download specific parts only:

```bash
python scripts/01_download_dataset.py --parts sparse pubtabnet
```

**Result**: `data/raw/` directory with 4 subdirectories (fintabnet, marketing, pubtabnet, sparse).

**Check**: each subdirectory should contain a `synthetic_data.jsonl` file and an `images/` folder.

---

### Step 2. Build the rotated-table benchmark

```bash
python scripts/02_build_benchmark.py
```

Takes 250 test images from each part, applies a deterministic rotation
(0°, 90°, 180°, 270° — evenly distributed with 250 samples per angle), and saves:
- Rotated images → `data/benchmark/images/`
- Metadata (GT HTML, angle, part) → `data/benchmark/benchmark_meta.jsonl`
- Statistics → `data/benchmark/benchmark_stats.json`

For a quick test (fewer images):

```bash
python scripts/02_build_benchmark.py --samples-per-part 10
```

**Result**: 1000 images in `data/benchmark/images/`, `benchmark_meta.jsonl` file.

**Check**: the script prints validation — it should say "Validation: PASSED",
exactly 250 samples for each angle and each part.

---

### Step 3. Evaluate VLM baselines (without rotation correction)

```bash
# TRivia-3B — specialized VLM for tables (OTSL → HTML format)
python scripts/03_evaluate_baselines.py --model trivia

# Qwen2.5-VL-3B — general-purpose VLM (direct HTML output)
python scripts/03_evaluate_baselines.py --model qwen
```

Each run evaluates the VLM on all 1000 benchmark images **as is** (rotated),
without orientation correction. This is the baseline — it shows how well the VLM handles rotated tables.

Models are downloaded automatically from Hugging Face on first run (~3 GB each).

Details:
- Inference is done one image at a time (`batch_size=1`) due to memory limits
- Progress is saved every 50 samples — you can stop and continue with `--resume`

For a quick test (first N samples):

```bash
python scripts/03_evaluate_baselines.py --model trivia --limit 10
```

**Result**: `results/baseline_trivia.json` and `results/baseline_qwen.json` files.

**Check**: the JSON file should contain 1000 entries (or `--limit`), each with the fields
`pred_html`, `gt_html`, `angle`, `part`, `inference_time`.

---

### Step 4. Train the rotation classifier

```bash
python scripts/04_train_rotation_classifier.py
```

Trains ResNet-18 (ImageNet pretrained) for 4-class rotation angle classification.
The SynthTabNet train split is used for training (it does not overlap with the benchmark).

Training:
- 5000 images × 4 parts × 4 rotations = 80000 training examples
- Adam, lr=1e-3, cosine annealing, early stopping (patience=3)

For a quick test:

```bash
python scripts/04_train_rotation_classifier.py --max-per-part 200 --epochs 3
```

Alternative architectures:

```bash
python scripts/04_train_rotation_classifier.py --arch resnet34
python scripts/04_train_rotation_classifier.py --arch efficientnet_b0
```

**Result**:
- Model weights → `models/rotation_classifier/best_model.pth`
- Training log → `models/rotation_classifier/training_log.json`
- Confusion matrix → `models/rotation_classifier/confusion_matrix.png`

**Check**: val accuracy should be > 95% (expected ~98-99%).
A trivial baseline (random guessing) = 25%.

---

### Step 5. Final pipeline (classifier + VLM)

```bash
# TRivia-3B with rotation correction
python scripts/05_run_pipeline.py --model trivia

# Qwen2.5-VL-3B with rotation correction
python scripts/05_run_pipeline.py --model qwen
```

Pipeline for each image:
1. ResNet-18 predicts the rotation angle
2. The image is corrected (rotated back)
3. The VLM receives the aligned image and generates table HTML

Supports `--resume` and `--limit` in the same way as step 3.

**Result**: `results/pipeline_trivia.json` and `results/pipeline_qwen.json` files.

**Check**: in the JSON file, each record contains additional fields
`predicted_rotation`, `rotation_confidence`, `rotation_correct`.

---

### Step 6. Compute metrics

```bash
# Metrics for each results file
python scripts/06_compute_metrics.py --predictions results/baseline_trivia.json
python scripts/06_compute_metrics.py --predictions results/baseline_qwen.json
python scripts/06_compute_metrics.py --predictions results/pipeline_trivia.json
python scripts/06_compute_metrics.py --predictions results/pipeline_qwen.json
```

For each file, computes:
- TEDS and TEDS-struct (mean, std, median, 95% CI)
- Breakdown by rotation angle (0°, 90°, 180°, 270°)
- Breakdown by dataset part (fintabnet, marketing, pubtabnet, sparse)
- Error analysis: error categories, worst-K samples

With visualization (boxplot, heatmap, histogram):

```bash
python scripts/06_compute_metrics.py --predictions results/baseline_trivia.json --visualize
```

**Result**: `results/metrics_<name>.json` file + plots in `results/plots/`.

---

### Step 7. Comparative analysis

```bash
python scripts/06_compute_metrics.py --compare
```

Compares all result files in `results/`:
- Summary table of TEDS/TEDS-struct for baseline vs pipeline
- Wilcoxon signed-rank test (statistical significance of the improvement)
- Cohen's d (effect size)
- Final report → `results/comparison.json`

**Result**: `results/comparison.json` and a table in stdout.

---

## Quick run (smoke test)

To verify that everything works without waiting for the full inference:

```bash
source venv/bin/activate

# Step 1 — download only one part
python scripts/01_download_dataset.py --parts sparse

# Step 2 — small benchmark
python scripts/02_build_benchmark.py --samples-per-part 10

# Step 3 — baseline on 5 images
python scripts/03_evaluate_baselines.py --model trivia --limit 5

# Step 4 — quick classifier training
python scripts/04_train_rotation_classifier.py --max-per-part 100 --epochs 2

# Step 5 — pipeline on 5 images
python scripts/05_run_pipeline.py --model trivia --limit 5

# Step 6 — metrics
python scripts/06_compute_metrics.py --predictions results/baseline_trivia.json
python scripts/06_compute_metrics.py --predictions results/pipeline_trivia.json

# Step 7 — comparison
python scripts/06_compute_metrics.py --compare
```

## Expected results

| Metric | Baseline (0°) | Baseline (90°/180°/270°) | Pipeline (all angles) |
|---------|--------------|--------------------------|-----------------------|
| TEDS | ~0.7-0.9 | ~0.0-0.3 | ~0.7-0.9 |
| TEDS-struct | ~0.8-0.95 | ~0.1-0.4 | ~0.8-0.95 |

**Expectation**: VLMs work well on 0° (correct orientation), but degrade sharply
on rotated images. The pipeline with rotation correction should restore
quality on rotated images to a level close to 0°.

## Models used

| Model | HuggingFace ID | Size | Purpose |
|--------|---------------|--------|------------|
| TRivia-3B | `opendatalab/TRivia-3B` | ~3 GB | VLM for tables (OTSL format) |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | ~3 GB | General-purpose VLM (HTML format) |
| ResNet-18 | torchvision (ImageNet pretrained) | ~45 MB | Rotation classifier |
