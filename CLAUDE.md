# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

MAVERIC is a multi-modal dataset curation system for vision-language models. The codebase follows a modular architecture:

- **`maveric/main.py`**: Main MAVERIC class providing high-level API for retrieval, quality control, and model customization
- **`maveric/core/`**: Base interfaces, exceptions, and abstract components
- **`maveric/retrieval/`**: Dataset retrieval and caching system with CLIP-based embedding
- **`maveric/quality/`**: Quality assessment metrics (visual, semantic, multimodal consistency)
- **`maveric/customization/`**: Model fine-tuning with filtered data
- **`maveric/interactive/`**: Jupyter widgets for threshold selection and quality dashboards
- **`maveric/visualization/`**: Data distribution plots and sample galleries
- **`maveric/datasets/`**: Dataset handlers for CIFAR, Elevater, and other datasets
- **`maveric/models/`**: CLIP model wrappers and factory patterns

## Development Commands

### Installation
```bash
# Development install
pip install -e ".[dev]"

# Install with docs dependencies
pip install -e ".[dev,docs]"
```

### Testing

#### Basic Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=maveric --cov-report=html

# Run specific test file
pytest tests/test_quality_metrics.py

# Run specific test
pytest tests/test_main.py::test_retrieve
```

#### Headless Environment Testing
For Docker, CI/CD, or remote environments without display:

```bash
# Set matplotlib backend and run tests
MPLBACKEND=Agg pytest

# Alternative: set in environment
export MPLBACKEND=Agg
pytest
```

#### Common Test Issues and Solutions

**Import Errors:**
- Missing `openai-clip`: `pip install openai-clip`
- Missing `torch`/`torchvision`: Install appropriate version for your system
- `libGL.so.1` errors: Use `opencv-python-headless` instead of `opencv-python`

**Matplotlib/Visualization Issues:**
- `seaborn` style not found: Updated to use 'default' style
- Display errors in headless environments: Set `MPLBACKEND=Agg`

**PIL API Changes:**
- `Image.BLUR` deprecated: Use `ImageFilter.BLUR` instead
- Import `ImageFilter` from PIL for filtering operations

**Test Environment Issues:**
- Use `tmp_path` fixture for temporary directories
- Mock external dependencies properly
- Import all required modules in test files (`torch`, `numpy`, etc.)

### Code Quality
```bash
# Format code
black maveric/ tests/

# Lint
flake8 maveric/ tests/

# Type checking
mypy maveric/
```

### CLI Usage
The package provides a CLI tool accessible via `maveric` command:

```bash
# Retrieve samples
maveric retrieve --source react-vl/react-retrieval-datasets --target cifar100 --num-samples 10000

# Apply quality control
maveric quality-control --input results.json --thresholds thresholds.json --output filtered.json

# Customize model
maveric customize --input filtered.json --model openai/clip-vit-base-patch32 --epochs 10 --output-dir ./models

# Visualize distributions
maveric visualize --input results.json --output-dir ./plots
```

## Configuration System

MAVERIC uses dataclass-based configuration in `config.py`:

- **MAVERICConfig**: Main system configuration (models, caching, quality thresholds, progress display)
- **TrainingConfig**: Model training parameters
- **ExperimentConfig**: Experiment management and tracking

Key configuration options:
- `enable_real_time_stats`: Show live download/cache statistics during retrieval (default: true)
- `clip_model`: CLIP model to use (default: "ViT-B/32")
- `cache_base_dir`: Directory for caching downloaded images and results
- `batch_size`: Processing batch size
- `retrieval_rotation_size`: Samples per file when saving results and training data (default: 1000)
- `quality_metrics`: List of quality metrics to compute

Configuration can be loaded from YAML/JSON files:
```python
config = MAVERICConfig.from_yaml('config.yaml')
maveric = MAVERIC.from_config_file('config.yaml')
```

## Key Components

### Quality Metrics (`maveric/quality/metrics/`)
- Visual metrics: resolution, sharpness, color diversity
- Semantic metrics: text-image consistency, feature richness
- Multimodal metrics: cross-modal alignment scores

### Retrieval System (`maveric/retrieval/`)
- CLIP-based embedding similarity matching
- Smart caching system for images and embeddings
- Dataset handlers for different source formats

### Interactive Dashboard (`maveric/interactive/`)
- Jupyter widget for real-time threshold tuning
- Quality distribution visualization
- Sample gallery with filtering

## Data Flow

1. **Retrieval**: Load source dataset → Generate CLIP embeddings → Match against target dataset embeddings
2. **Quality Assessment**: Apply visual/semantic metrics → Score each sample
3. **Filtering**: Apply thresholds → Balance dataset → Export filtered results
4. **Customization**: Fine-tune model on filtered data → Evaluate performance

## Testing Strategy

- Unit tests for each component in `tests/`
- Configuration fixtures in `tests/conftest.py`
- Quality metrics validation with known datasets
- Integration tests for full pipeline workflows

## Cache Management

Images and embeddings are cached in configurable directories:
- Image cache: JPEG compressed images for fast loading
- Embedding cache: Precomputed CLIP embeddings
- Results cache: Serialized retrieval and quality results