<p align="center">
  <img src="docs/maveric_logo.svg" alt="MAVERIC" width="800" />
</p>

# MAVERIC: Mahalanobis-based Adaptive Vision-language Efficient Retrieval with Integrated Curation

MAVERIC is a quality-driven dataset curation system for vision-language models. It retrieves training samples from large web-crawled datasets, scores them with multi-modal quality metrics, and fine-tunes CLIP models on the curated data — following the [REACT benchmark](https://react-vl.github.io/) protocol across all 20 ELEVATER datasets.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Configuration](#configuration)
- [Experiment Scripts](#experiment-scripts)
- [Unified Training](#unified-training)
- [Interactive GUI](#interactive-gui)
- [Datasets](#datasets)
- [CLI Reference](#cli-reference)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

MAVERIC implements a 4-stage pipeline:

1. **Retrieve** — Download candidate images from source datasets (LAION, etc.) using CLIP embedding similarity to a target dataset's reference images
2. **Curate** — Score each sample with visual, semantic, and multimodal quality metrics; filter interactively via a Jupyter GUI
3. **Customize** — Fine-tune CLIP's vision encoder on curated data with regularization to prevent catastrophic forgetting
4. **Evaluate** — Assess the customized model against baseline zero-shot CLIP per ELEVATER dataset

Key design decisions:
- **REACT-style evaluation**: Dataset-specific text templates with template ensembling for reproducible benchmarks
- **Locked-text tuning**: Only the vision encoder is fine-tuned; the text encoder is frozen
- **MSE regularization**: Prevents the vision encoder from drifting too far from its pre-trained weights
- **Smart caching**: Cross-dataset sample metadata cache (v3) stores CLIP embeddings to eliminate redundant inference on subsequent runs

---

## Pipeline

```
Source Dataset (LAION/CC3M/...)
        │
        ▼
┌─────────────────────┐
│  01_data_retrieval  │  CLIP similarity matching → raw JSON with quality scores
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  02_data_curation   │  Interactive GUI thresholds → filtered training JSON
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

# Caching
enable_image_cache: true
enable_sample_cache: true            # Cross-dataset CLIP embedding cache (v3)
sample_cache_version: 3

# Quality metric weights
metric_weights:
  img2img: 0.25
  txt2txt: 0.25
  img2txt: 0.25
  txt2img: 0.25

# Training
training:
  epochs: 50
  learning_rate: 0.0000003
  weight_decay: 0.05
  regularization_weight: 0.7        # MSE regularization (prevents catastrophic forgetting)
  use_augmentation: true
  augmentation_strength: 3
  augmentation_magnitude: 12
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clip_value: 0.5
  use_domain_adaptation: false       # Enable per-dataset domain simulation
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

```bash
python experiments/01_data_retrieval.py \
    --config experiments/maveric_config.yaml \
    --dataset cifar10 \
    --output ./maveric_experiments/cifar10/raw/

# With EfficientNet quality scores (slower but more comprehensive)
python experiments/01_data_retrieval.py \
    --config experiments/maveric_config.yaml \
    --dataset cifar10 \
    --enable-efficientnet
```

**Output**: Rotation JSON files with per-sample quality scores:
- Visual: `resolution_score`, `sharpness_score`, `color_score`
- Semantic: `text_quality_score`, `caption_length_score`
- Multimodal: `weighted_class_score`, `consistency`, per-class `Class_{name}_img2img`, etc.

**Caching**: Sample metadata (including CLIP embeddings) cached at `{cache_dir}/sample_metadata_cache/`. Second retrieval on the same source is ~85% faster.

---

### 02_data_curation.py — Quality Control

```bash
python experiments/02_data_curation.py \
    --input-dir ./maveric_experiments/cifar10/raw/ \
    --dataset-name cifar10 \
    --config experiments/maveric_config.yaml \
    --output ./maveric_experiments/cifar10/curated/
```

Applies quality thresholds and balances the dataset. For interactive threshold selection, use the [GUI](#interactive-gui) instead.

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
3. Trains vision encoder with RandAugment + optional domain adaptation
4. Evaluates per epoch using REACT-style template ensembling
5. Saves `best_model.pth` checkpoint

---

### 04_results_analysis.py — Analysis & Visualization

```bash
python experiments/04_results_analysis.py \
    --input-dir ./maveric_experiments/ \
    --output-dir ./maveric_experiments/analysis/
```

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
```

Compares baseline zero-shot CLIP vs. customized model per dataset using REACT-style template ensembling.

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
| **Metric Weights** | Adjust img2img / txt2txt / img2txt / txt2img weights with live preview |
| **Quality Thresholds** | Set per-metric thresholds; see sample counts update in real time |
| **Mahalanobis Filter** | Joint filtering on `weighted_class_score` + `consistency` via distance from ideal point |
| **EfficientNet Prediction** | Filter by ImageNet class predictions (when EfficientNet scores available) |
| **Balance Settings** | Undersample/oversample to target class balance; sort by consistency or weighted score |

### Mahalanobis Filter

The Mahalanobis filter jointly optimizes two quality axes instead of independent thresholds:

- **Global mode**: Filter all classes at once
- **Class-based mode**: Filter each class individually, accumulate results
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
- 10×10 image grids organized by class for manual inspection (`curationResults/`)

---

## Datasets

MAVERIC supports all 20 official ELEVATER benchmark datasets:

### Torchvision-based (7) — auto-download

| Dataset | Classes | Notes |
|---------|---------|-------|
| CIFAR-10 | 10 | |
| CIFAR-100 | 100 | Alphabetically ordered classes |
| Country211 | 211 | |
| EuroSAT | 10 | Satellite imagery |
| GTSRB | 43 | Traffic signs |
| Oxford Flowers102 | 102 | |
| Oxford Pets | 37 | |

### File-based (13) — manual test data required

| Dataset | Classes | Notes |
|---------|---------|-------|
| Caltech101 | 102 | Includes `background_google` |
| DTD | 47 | Texture recognition |
| FER2013 | 7 | Facial expressions; list-based class names |
| FGVCAircraft | 100 | Migrated from torchvision Feb 2026 |
| Food101 | 101 | Migrated from torchvision Feb 2026 |
| HatefulMemes | 2 | |
| Kitti Distance | 4 | |
| MNIST | 10 | |
| PatchCamelyon | 2 | Lymph node classification |
| RenderedSST2 | 2 | Sentiment |
| RESISC45 | 45 | Remote sensing |
| StanfordCars | 196 | |
| VOC2007 | 20 | |

For file-based dataset setup instructions see [docs/newfeatures/FILE_BASED_DATASETS_GUIDE.md](docs/newfeatures/FILE_BASED_DATASETS_GUIDE.md).

Expected directory structure for file-based datasets:
```
{results_dir}/{dataset_name}/
├── train/
│   ├── class_name_1/
│   │   ├── image001.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class_name_1/
    └── ...
```

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
│   └── utils/                       # CLI, I/O, logging
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
