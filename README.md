<p align="center">
  <img src="docs/maveric_logo.svg" alt="MAVERIC" width="800" />
</p>

# MAVERIC: Mahalanobis-based Adaptive Vision-language Efficient Retrieval with Integrated Curation

MAVERIC is a **curation layer** for customizing vision-language models. It does not perform its own web-scale retrieval: for each ELEVATER task it takes image-text pairs from [REACT](https://react-vl.github.io/)'s released per-task retrieval output, scores every pair with multimodal quality metrics, keeps a small class-balanced subset via per-class Mahalanobis selection, and fine-tunes CLIP ViT-B/32 (locked-text) on that subset. It covers all 20 ELEVATER datasets.

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Configuration](#configuration)
- [Experiment Scripts](#experiment-scripts)
- [Consistency Analysis Tools](#consistency-analysis-tools)
- [Unified Training](#unified-training)
- [Interactive GUI](#interactive-gui)
- [Datasets](#datasets)
- [CLI Reference](#cli-reference)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

MAVERIC implements a 4-stage pipeline:

1. **Score**: Load REACT's retrieved image-text pairs for a target task and compute four directional CLIP similarities per class, plus visual and semantic quality metrics
2. **Curate**: Select a fixed, class-balanced budget of samples per class by Mahalanobis distance to an ideal point in the 2-D quality space `[q_avg, q_cons]`
3. **Customize**: Fine-tune CLIP's image encoder on the curated set with a cross-entropy loss against frozen class-prompt prototypes, plus MSE weight regularization
4. **Evaluate**: Compare the customized model against zero-shot CLIP using each dataset's official ELEVATER metric

Key design decisions:
- **Reference-guided curation**: Each class has 10 reference images (from the task's training split) and the official ELEVATER prompt templates as text references
- **Joint 2-D selection**: Samples are ranked by Mahalanobis distance on `[q_avg, q_cons]`. There are no per-metric thresholds and no tuned metric weights
- **Locked-text tuning**: Only the image encoder is fine-tuned; the text encoder stays frozen
- **MSE regularization**: Keeps the image encoder close to its pre-trained weights to limit catastrophic forgetting
- **ELEVATER-matched evaluation**: Template ensembling and per-dataset metrics (accuracy, mean-per-class, ROC AUC, 11-point mAP)
- **Smart caching**: A cross-dataset sample metadata cache (v3) stores CLIP embeddings, so later runs skip redundant inference

---

## Method

This section matches the revised manuscript (Sects. 3.3–3.6, Table 1, Appendix A).

### 1. Input data

For each ELEVATER task, MAVERIC takes the **first 500,000 image-text pairs** of REACT's per-task retrieval output (`react-vl/react-retrieval-datasets`). It keeps their stored row order and does no shuffling, de-duplication or re-ranking. Each pair keeps the class label of the query that retrieved it.

> The shipped config sets `elevater.retrieval.num_samples: 10000000`. To reproduce the paper, set it to `500000`.

### 2. Quality scoring

For every class `c`:

- **Image references** `R_c`: 10 images sampled without replacement from the task's training split (seed 42; `n_reference_images: 10`)
- **Text references** `P_c`: the official ELEVATER prompt templates (`template_map` in `vision_benchmark/datasets/prompts.py` of the ELEVATER toolkit) filled in with the class name

Each retrieved pair `(I, T)` gets four directional cosine similarities in CLIP space. Each one is the **maximum over the references** ([retriever.py](maveric/retrieval/retriever.py)):

| Score | Output field | Compares |
|---|---|---|
| `q_i2i` | `Class_{c}_img2img` | retrieved image vs. reference images |
| `q_t2t` | `Class_{c}_txt2txt` | retrieved caption vs. class prompts |
| `q_i2t` | `Class_{c}_img2txt` | retrieved image vs. class prompts |
| `q_t2i` | `Class_{c}_txt2img` | retrieved caption vs. reference images |

Two composite scores form the 2-D quality vector `q = [q_avg, q_cons]`:

- `q_avg = mean(q_i2i, q_t2t, q_i2t, q_t2i)`, stored as `weighted_class_score` (and `hybrid_score`). It uses equal weights: keep `metric_weights` at `0.25` each
- `q_cons = 1 - std(q_i2i, q_t2t, q_i2t, q_t2i)`, stored as `consistency`, measures agreement across the four directions

Optionally, `consistency` can be computed on per-class, per-metric **z-score normalized** similarities (`consistency_normalization: "zscore"`). `q_avg` always uses the raw similarities. See [Consistency Analysis Tools](#consistency-analysis-tools).

### 3. Mahalanobis curation

For each class separately:

1. **Ideal point** `mu* = [P_{p_avg}(q_avg), P_{p_cons}(q_cons)]`, where `P_p` is a percentile over the class's candidates:
   - `|C| <= 10` classes (coarse-grained tasks): `p_avg = 99.99`, `p_cons = 25`
   - `|C| > 10` classes (fine-grained tasks): `p_avg = 99.99`, `p_cons = 99.99`
2. **Covariance** of the class's quality vectors, regularized as `Sigma + 1e-6 * I`
3. **Mahalanobis distance** from every candidate to `mu*`
4. **Rank-based selection**: sort by ascending distance and keep the closest `n_tau` samples

**Budget.** Each task has a total curated-set size `N_tau` of 1,000, 2,000, 4,000 or 6,000 (per-dataset values are in Table A1 of the paper). The per-class budget is `n_tau = floor(N_tau / K)`. The remaining `N_tau - K * n_tau` slots go one each to the lowest class indices, so exactly `N_tau` samples are kept. Across the 20 tasks this gives 88,000 samples.

In the code, this is the **Mahalanobis Filter** tab of the [Interactive GUI](#interactive-gui) in **Class-Based** mode. Set **Weighted %ile** to `p_avg`, **Consistency %ile** to `p_cons` and **Keep Count** to `n_tau`. **ALL** applies the same Keep Count to every class, so any remainder slots have to be added per class.

MAVERIC does **not** use independent per-metric thresholding or a weighted combination of the four similarities. Both came from earlier versions of this work. The *Metric Weights* and *Quality Thresholds* GUI tabs remain for exploration only.

### 4. Training objective (locked-text)

- The **text encoder is frozen**; only the image encoder is updated
- Training uses a **cross-entropy loss against fixed per-class text prototypes**. Each prototype is the L2-normalized mean text embedding of the class's prompt templates, and targets are (image, class prototype) pairs (`training.text_source: "labels"`)
- This is **not** CLIP's contrastive image-caption loss. Retrieved captions are used only during curation, through the quality scores, and never as training targets

Hyperparameters, fixed for all 20 tasks:

| Parameter | Value | Config key (`training:`) |
|---|---|---|
| Batch size | 32 | `batch_size` (top level) |
| Epochs | 30 | `epochs` |
| Learning rate | 1e-7 | `learning_rate` |
| Warm-up steps | 200 | `warmup_steps` |
| Optimizer | AdamW, weight decay 0.05 | `optimizer`, `weight_decay` |
| Gradient clip | 0.5 | `gradient_clip_value` |
| Regularization weight (MSE) | 0.9 | `regularization_weight` |
| RandAugment | N=2, M=8 | `augmentation_strength`, `augmentation_magnitude` |
| Hardware | single NVIDIA T4 GPU | |

> The shipped `experiments/maveric_config.yaml` sets `epochs: 50`. To reproduce the paper, set it to `30`.

### 5. Model selection and evaluation

- The curated set is split **80/20 (stratified, seed 42)** into training and validation subsets. In the code, `validation_method: "stratified_kfold"` with `validation_k_folds: 5` uses the first fold
- The checkpoint with the **highest validation accuracy** is kept (`best_model.pth`). The test set is evaluated **once**, for that checkpoint only
- Evaluation uses ELEVATER's zero-shot interface: cosine similarity to class-prompt embeddings with template ensembling. It reports the official per-dataset metric (top-1 accuracy, mean-per-class accuracy, ROC AUC for HatefulMemes, 11-point mAP for VOC2007)
- Labeled reference images are used during curation. The protocol is therefore **reference-guided curation followed by zero-shot-style evaluation**, not a purely zero-shot pipeline

---

## Pipeline

```
REACT per-task retrieval output (react-vl/react-retrieval-datasets)
        │
        ▼
┌─────────────────────┐
│  01_data_retrieval  │  Per-class q_i2i/q_t2t/q_i2t/q_t2i, q_avg, q_cons → raw JSON
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  02_data_curation   │  Per-class Mahalanobis selection (GUI) → curated JSON
└─────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  03_model_customization  │  Fine-tune CLIP vision encoder → best_model.pth
└──────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  04_results_analysis│  Plots, tables, markdown report
└─────────────────────┘
```

For multi-dataset training, `03_model_customization.py --unified-training` combines all datasets into a single training run, evaluated by `05_unified_evaluation.py`.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional but recommended)

### Standard Install

```bash
git clone <this-repository-url>
cd maveric
pip install -r requirements.txt
pip install -e ".[dev]"
```

### System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y $(grep -v "^#" system-requirements.txt | xargs)
```

Required system packages: `libgl1-mesa-glx`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `libgomp1`

### Headless / Docker / CI

```bash
pip install opencv-python-headless
export MPLBACKEND=Agg
```

### Google Colab Setup

```bash
python experiments/00_setup.py --config experiments/maveric_config.yaml
```

This script mounts Google Drive, installs dependencies, sets environment variables, and validates the installation.

### Common Issues

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: clip` | `pip install openai-clip` |
| `libGL.so.1` not found | Use `opencv-python-headless` |
| Matplotlib display error | `export MPLBACKEND=Agg` |
| `transformers` breaking changes | Pin with `pip install "transformers>=4.20.0,<5.0.0"` |

---

## Configuration

All behaviour is controlled via `experiments/maveric_config.yaml`. Load it programmatically:

```python
from maveric import MAVERIC
maveric = MAVERIC.from_config_file('experiments/maveric_config.yaml')
```

### Key Parameters

```yaml
# Paths
maveric_base_dir: "/content/drive/MyDrive/MAVERIC"
cache_base_dir:   "/content/drive/MyDrive/MAVERIC/maveric_cache"
results_dir:      "/content/drive/MyDrive/MAVERIC/maveric_experiments"

# Model
clip_model: "ViT-B/32"   # also: ViT-B/16, ViT-L/14, ViT-L/14@336px
device: "cuda"
batch_size: 32

# Retrieval
enable_target_class_quality: false   # Disable EfficientNet for 50-70% faster retrieval
n_reference_images: 10
request_timeout: 1
max_retries: 0

# Scoring mode for class similarity: "clip" (default, multi-modal) or
# "hu_moments" (shape-based, rotation/scale/translation-invariant alternative)
scoring_mode: "clip"                 # paper uses "clip"

# Caching
enable_image_cache: true
enable_sample_cache: true            # Cross-dataset CLIP embedding cache (v3)
sample_cache_version: 3

# Consistency normalization: "none" (default) or "zscore"
# (per-class, per-metric z-scores before computing q_cons)
consistency_normalization: "none"

# Metric weights for q_avg: keep equal (0.25 each) to match the paper
metric_weights:
  img2img: 0.25
  txt2txt: 0.25
  img2txt: 0.25
  txt2img: 0.25

# Per-dataset official ELEVATER evaluation metric (default: accuracy).
# Applied by 03_model_customization.py and 05_unified_evaluation.py.
evaluation_metrics:
  caltech101: "mean_per_class"
  oxford_pets: "mean_per_class"
  fgvc_aircraft: "mean_per_class"
  oxford_flowers102: "mean_per_class"
  hateful_memes: "roc_auc"
  voc2007: "voc11_map"

# Save 10x10 image grids to curationResults/ when clicking "Save Data" in
# the interactive GUI. Disabled by default for faster curation.
save_grid_visualization: false

# Training (see Method, section 4, for the paper setting)
training:
  epochs: 50                         # paper: 30
  learning_rate: 0.0000001
  warmup_steps: 200
  weight_decay: 0.05
  regularization_weight: 0.9        # MSE regularization (prevents catastrophic forgetting)
  use_augmentation: true
  augmentation_strength: 2
  augmentation_magnitude: 8
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clip_value: 0.5
  use_domain_adaptation: false       # Enable per-dataset domain simulation
  validation_method: "stratified_kfold"  # first of 5 folds = stratified 80/20 split
  validation_k_folds: 5
  text_source: "labels"              # "labels": CE vs. class-prompt prototypes (paper)
                                      # "captions": experimental InfoNCE on captions, not used in the paper
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MAVERIC_BASE_DIR` | Root directory for all MAVERIC files |
| `MAVERIC_CACHE_DIR` | Image and embedding cache |
| `MAVERIC_RESULTS_DIR` | Experiment results and logs |
| `MAVERIC_CONFIG_PATH` | Path to configuration file |
| `HF_HOME` | Hugging Face model cache |

---

## Experiment Scripts

Run scripts in order from the `experiments/` directory:

### 00_setup.py — Environment Setup

```bash
python experiments/00_setup.py --config experiments/maveric_config.yaml
```

Sets up the environment for Colab or local use: mounts Drive, installs packages, creates directories.

---

### 01_data_retrieval.py — Data Retrieval

This script is interactive: it prompts you to pick a target dataset (by number or name) and a starting index after launch. Output location is derived from `results_dir` in the config (`{results_dir}/{dataset}/raw/`), not a CLI flag.

```bash
python experiments/01_data_retrieval.py --config experiments/maveric_config.yaml
# → 🎯 Dataset Selection: enter a number or name (e.g. "cifar10") when prompted
# → 📍 Starting Index Selection: enter 0 to start from the beginning

# With EfficientNet quality scores (slower but more comprehensive)
python experiments/01_data_retrieval.py \
    --config experiments/maveric_config.yaml \
    --enable-efficientnet

# Other flags: --dataset-id <int> (output filename suffix, default 1),
# --debug-timing (per-sample timing breakdown for performance debugging)
```

**Output**: Rotation JSON files with per-sample quality scores:
- Visual: `resolution_score`, `sharpness_score`, `color_score`
- Semantic: `text_quality_score`, `caption_length_score`
- Multimodal: per-class `Class_{name}_img2img` / `_txt2txt` / `_img2txt` / `_txt2img`, `_hybrid_score` (= `q_avg`), `_consistency` (= `q_cons`); the curated output carries the assigned class's `weighted_class_score` and `consistency`

**Caching**: Sample metadata (including CLIP embeddings) cached at `{cache_dir}/sample_metadata_cache/`. Second retrieval on the same source is ~85% faster.

---

### 02_data_curation.py — Quality Control

```bash
python experiments/02_data_curation.py \
    --input-dir ./maveric_experiments/cifar10/raw/ \
    --dataset-name cifar10 \
    --config experiments/maveric_config.yaml \
    --output-dir ./maveric_experiments/cifar10/curated/ \
    --balance-strategy median   # none, median, min, or max
```

Applies threshold-based filtering and balancing, which is the legacy, non-paper path. To run the paper's per-class Mahalanobis curation, use the **Mahalanobis Filter** tab of the [GUI](#interactive-gui) (see [Method, section 3](#3-mahalanobis-curation)). `--output-dir`/`-o` defaults to `{results_dir}/{dataset-name}` from the config if omitted.

---

### 03_model_customization.py — Model Fine-Tuning

**Single dataset:**
```bash
python experiments/03_model_customization.py \
    --input ./maveric_experiments/cifar10/curated/ \
    --config experiments/maveric_config.yaml \
    --output-dir ./maveric_experiments/cifar10/models/

# Save augmented sample grids for inspection
python experiments/03_model_customization.py \
    --input ./maveric_experiments/cifar10/curated/ \
    --config experiments/maveric_config.yaml \
    --save-augmented-grids
```

**Unified training across all datasets:**
```bash
python experiments/03_model_customization.py \
    --input ./maveric_experiments/unified_training_data/ \
    --config experiments/maveric_config.yaml \
    --unified-training \
    --output-dir ./maveric_experiments/unified_training/models/
```

The fine-tuning process:
1. Loads curated training JSON files
2. Wraps CLIP in `CustomizedCLIP` (locks text encoder, enables MSE regularization)
3. Splits the curated set 80/20 (stratified, seed 42) into train/validation
4. Trains the image encoder with cross-entropy against frozen class-prompt prototypes, RandAugment and optional domain adaptation
5. Evaluates baseline and per-epoch accuracy using REACT-style template ensembling and the dataset's official ELEVATER metric (`evaluation_metrics` in config — see [Datasets](#datasets))
6. Saves the checkpoint with the highest validation accuracy as `best_model.pth`

`--epochs`, `--learning-rate`, and `--batch-size` override the corresponding config values. Unified training also accepts `--max-samples-per-dataset` to cap per-dataset sample count when balancing across datasets.

---

### 04_results_analysis.py — Analysis & Visualization

This script takes **no command-line arguments**. It reads its config path from the `MAVERIC_CONFIG_PATH` environment variable (falling back to a hardcoded Colab path if unset — set this variable rather than relying on the default), then reads `results_dir` from that config.

```bash
export MAVERIC_CONFIG_PATH=experiments/maveric_config.yaml
python experiments/04_results_analysis.py
```

> **Note**: this script expects `{results_dir}/elevater_experiment_summary.json`, which is produced by a batch multi-dataset experiment runner rather than by the `01`→`02`→`03` single-dataset flow shown above. If you're running datasets individually, this summary file won't exist yet — check for it before relying on this script, or adapt it to read your own experiment logs.

Generates comparison plots, accuracy tables, and a markdown report.

---

### 05_unified_evaluation.py — Unified Model Evaluation

```bash
# Evaluate unified model on all datasets
python experiments/05_unified_evaluation.py \
    --checkpoint ./maveric_experiments/unified_training/models/best_model.pth \
    --config experiments/maveric_config.yaml

# Skip baseline evaluation (faster)
python experiments/05_unified_evaluation.py \
    --checkpoint ./maveric_experiments/unified_training/models/best_model.pth \
    --no-baseline

# Evaluate on specific datasets only
python experiments/05_unified_evaluation.py \
    --checkpoint ./maveric_experiments/unified_training/models/best_model.pth \
    --datasets cifar10 cifar100 oxford_pets

# Also save per-class accuracy breakdown
python experiments/05_unified_evaluation.py \
    --checkpoint ./maveric_experiments/unified_training/models/best_model.pth \
    --detailed
```

Compares baseline zero-shot CLIP vs. customized model per dataset using REACT-style template ensembling and each dataset's official ELEVATER metric (same `evaluation_metrics` config mapping used by `03_model_customization.py` — mean-per-class, ROC AUC, or VOC 11-point mAP where applicable, top-1 accuracy otherwise). `--config` defaults to `experiments/maveric_config.yaml` if omitted.

---

## Consistency Analysis Tools

[maveric/utils/consistency_analysis.py](maveric/utils/consistency_analysis.py) and [maveric/utils/multi_file_consistency_analysis.py](maveric/utils/multi_file_consistency_analysis.py) test whether `q_cons` carries real multimodal signal, and optionally recompute it on z-score normalized similarities.

**Null-model permutation test** (single file). This checks whether the correlation between `q_avg` and `q_cons` goes beyond what the shared similarities produce mechanically:

```bash
python -m maveric.utils.consistency_analysis \
    --data results/cifar10/curated/data.json \
    --normalization none \
    --iterations 1000 \
    --output report.txt
```

**Null-model test across rotation files:**

```bash
python -m maveric.utils.multi_file_consistency_analysis \
    --data-dir results/cifar10/raw \
    --normalization zscore \
    --output report.txt
```

**Z-score normalize a whole raw retrieval directory.** Per-class statistics are pooled across all rotation files as one population, and output files mirror the input filenames:

```bash
python -m maveric.utils.consistency_analysis \
    --input-dir results/hateful_memes/raw \
    --output-dir results/hateful_memes/raw_zscore
```

Use the directory mode instead of per-file `--apply-zscore`. A single rotation file holds only part of each class, so its mean and standard deviation are biased. Z-scoring changes only `consistency`; `weighted_class_score` stays on the raw scale.

---

## Unified Training

Unified training combines curated data from multiple ELEVATER datasets into a single fine-tuning session, producing one model evaluated across all datasets.

### Directory Structure

```
maveric_experiments/
├── cifar10/
│   ├── images/          # Locally cached images (fast access)
│   └── *training*maveric*.json
├── cifar100/
│   ├── images/
│   └── *training*maveric*.json
├── oxford_pets/
│   ├── images/
│   └── *training*maveric*.json
└── unified_training_data/   # Symlinks or copies for unified training
    ├── cifar10/ → ../cifar10/
    ├── cifar100/ → ../cifar100/
    └── ...
```

The unified trainer automatically:
- Builds a merged class space across all datasets (e.g. 1,151 classes for 20 datasets)
- Indexes dataset-specific `images/` folders for fast local validation (avoids slow network cache)
- Applies dataset-specific domain adaptation transforms (blur, JPEG compression, resolution scaling)

---

## Interactive GUI

For interactive data curation in Jupyter or Google Colab:

```python
from maveric.visualization import start_interactive_gui

gui = start_interactive_gui('cifar10', config_file='experiments/maveric_config.yaml')
```

### GUI Tabs

| Tab | Purpose |
|-----|---------|
| **Metric Weights** | Adjust img2img / txt2txt / img2txt / txt2img weights (exploratory; paper uses equal weights) |
| **Quality Thresholds** | Per-metric thresholds (legacy/exploratory; not part of the paper protocol) |
| **Mahalanobis Filter** | Paper's curation step: rank by Mahalanobis distance to an ideal point on `[q_avg, q_cons]` |
| **EfficientNet Prediction** | Filter by ImageNet class predictions (when EfficientNet scores available) |
| **Balance Settings** | Undersample/oversample to target class balance; sort by consistency or weighted score |

### Mahalanobis Filter

The Mahalanobis filter ranks samples jointly on `q_avg` (`weighted_class_score`) and `q_cons` (`consistency`). The ideal point is set by two percentiles, the covariance is regularized with `1e-6 * I`, and the closest samples are kept.

- **Class-based mode** (paper protocol): Filter each class individually, accumulate results
- **Global mode**: Filter all classes at once
- **Batch ALL**: Process all classes with same parameters in one click
- **Keep Count**: Enter exact sample count (e.g. 350) instead of a percentage

```python
# Programmatic usage
gui.apply_thresholds()    # Apply Tab 2 thresholds
gui.apply_balance()       # Apply Tab 5 balancing
gui.save_data()           # Export to JSON + auto-generate image grids
```

### Saving Data

Clicking **Save Data** exports:
- Training JSON files (rotation files, 1000 samples each)
- 10×10 image grids organized by class for manual inspection (`curationResults/`) — **disabled by default** for faster curation; set `save_grid_visualization: true` in the config to enable

---

## Datasets

MAVERIC supports all 20 official ELEVATER benchmark datasets:

### Torchvision-based (6) — auto-download

| Dataset | Classes | Notes |
|---------|---------|-------|
| CIFAR-10 | 10 | |
| CIFAR-100 | 100 | Alphabetically ordered classes |
| DTD | 47 | Texture recognition |
| GTSRB | 43 | Traffic signs |
| Oxford Flowers102 | 102 | |
| Oxford Pets | 37 | |

### File-based (14) — manual test data required

| Dataset | Classes | Notes |
|---------|---------|-------|
| Caltech101 | 102 | Includes `background_google` |
| Country211 | 211 | |
| EuroSAT | 10 | Satellite imagery |
| FER2013 | 7 | Facial expressions; list-based class names/synonyms |
| FGVCAircraft | 100 | Migrated from torchvision Feb 2026 |
| Food101 | 101 | Migrated from torchvision Feb 2026 |
| HatefulMemes | 2 | Evaluated with ROC AUC |
| Kitti Distance | 4 | |
| MNIST | 10 | |
| PatchCamelyon | 2 | Lymph node classification |
| RenderedSST2 | 2 | Sentiment |
| RESISC45 | 45 | Remote sensing |
| StanfordCars | 196 | |
| VOC2007 | 20 | Multi-label; evaluated with VOC 11-point mAP |

For file-based dataset setup instructions see [docs/newfeatures/FILE_BASED_DATASETS_GUIDE.md](docs/newfeatures/FILE_BASED_DATASETS_GUIDE.md).

Expected directory structure for file-based datasets — test data is read from `{cache_base_dir}/{dataset_name}/datasets/elevater/{dataset_name}/`, **not** `results_dir`:
```
{cache_base_dir}/{dataset_name}/datasets/elevater/{dataset_name}/
├── train/                # Optional: used for reference sampling during retrieval
│   ├── class_name_1/
│   │   ├── image001.jpg
│   │   └── ...
│   └── ...
└── test/                 # Required: used for baseline/customized model evaluation
    ├── class_name_1/
    └── ...
```

Most datasets use a per-class official evaluation metric instead of plain top-1 accuracy (configured via `evaluation_metrics` in `maveric_config.yaml` and applied consistently by `03_model_customization.py` and `05_unified_evaluation.py`): `caltech101`, `oxford_pets`, `fgvc_aircraft`, `oxford_flowers102` use mean-per-class (balanced) accuracy; `hateful_memes` uses ROC AUC; `voc2007` uses 11-point interpolated mAP over multi-hot labels; all other datasets default to standard top-1 accuracy.

---

## CLI Reference

```bash
# Retrieve samples
maveric retrieve \
    --source react-vl/react-retrieval-datasets \
    --target cifar10 \
    --num-samples 100000 \
    --config experiments/maveric_config.yaml

# Quality control
maveric quality-control \
    --input results.json \
    --thresholds thresholds.json \
    --balance median \
    --output filtered.json

# Fine-tune model
maveric customize \
    --input filtered.json \
    --model openai/clip-vit-base-patch32 \
    --epochs 20 \
    --output-dir ./models

# Visualize distributions
maveric visualize \
    --input results.json \
    --output-dir ./plots
```

---

## Reproducibility

One seed value, **42**, is used for every source of randomness. That covers reference sampling, the train/validation split, Python/NumPy/PyTorch during customization, the random-selection controls, and the permutation null model. Pass `--seed 42` to the consistency analysis tools, whose CLI default is `0`.

Checklist for reproducing the paper with `experiments/maveric_config.yaml`:

- `clip_model: "ViT-B/32"`, `scoring_mode: "clip"`, `n_reference_images: 10`, `metric_weights` all `0.25`
- `elevater.retrieval.num_samples: 500000`
- Class-based Mahalanobis curation with the percentiles and per-task budget from [Method, section 3](#3-mahalanobis-curation)
- `training.epochs: 30`, plus the other values in the [Method, section 4](#4-training-objective-locked-text) table; `training.text_source: "labels"`

---

## Testing

```bash
# Run all tests
pytest

# Headless environments (Docker, CI, remote servers)
MPLBACKEND=Agg pytest

# With coverage
MPLBACKEND=Agg pytest --cov=maveric --cov-report=html

# Specific test file
pytest tests/test_quality_metrics.py -v
```

Test files are located in [tests/](tests/).

---

## Project Structure

```
maveric/
├── experiments/
│   ├── 00_setup.py                  # Environment setup
│   ├── 01_data_retrieval.py         # Retrieval stage
│   ├── 02_data_curation.py          # Quality control stage
│   ├── 03_model_customization.py    # Fine-tuning stage
│   ├── 04_results_analysis.py       # Analysis & visualization
│   ├── 05_unified_evaluation.py     # Unified model evaluation
│   └── maveric_config.yaml          # Configuration
├── maveric/
│   ├── main.py                      # MAVERIC class (high-level API)
│   ├── config.py                    # MAVERICConfig, TrainingConfig
│   ├── core/                        # Base classes, interfaces, exceptions
│   ├── retrieval/                   # Retrieval engine + caching
│   ├── datasets/                    # ELEVATER dataset handlers
│   ├── quality/                     # Quality metrics & filtering
│   ├── customization/               # Model fine-tuning & evaluation
│   │   ├── model_customizer.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── unified_training.py
│   ├── models/                      # CLIP wrappers
│   ├── visualization/               # Interactive GUI & plots
│   └── utils/                       # CLI, I/O, logging, balancing, consistency analysis
├── tests/                           # Test suite
├── docs/
│   ├── bugfixes/                    # Bug fix documentation
│   ├── newfeatures/                 # Feature documentation
│   └── maveric_pipeline.svg         # Architecture diagram
├── requirements.txt
├── system-requirements.txt
└── setup.py
```

---

## Citation

If you use MAVERIC in your research, please cite:

```bibtex
@software{maveric2025,
  title   = {MAVERIC: Mahalanobis-based Adaptive Vision-language Efficient Retrieval with Integrated Curation},
  author  = {Anonymous Author(s)},
  year    = {2025},
  note    = {Under double-blind review}
}
```

*(Citation details will be updated with author names and publication venue upon acceptance.)*

---

## License

MIT License — see [LICENSE](LICENSE) for details.
