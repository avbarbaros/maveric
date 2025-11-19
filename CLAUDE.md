# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference - Recent Updates

### November 5, 2025 - CLIP Embedding Caching (LATEST)
- **Enhanced sample caching**: CLIP embeddings now cached alongside metrics
  - **Cache version upgraded to v3**: Includes CLIP image/text embeddings (base64 encoded)
  - **Performance Impact**: 80-95% speedup for subsequent dataset retrievals (vs 60-85% without embeddings)
  - **Storage**: ~17KB per sample (~500 bytes metrics + ~16KB embeddings)
  - **Total for 270K samples**: ~4.5GB (increased from ~135MB without embeddings)
  - **Configuration**: `enable_sample_cache: true` (default), `sample_cache_version: 3`
  - **Key benefit**: Eliminates CLIP inference on cache hits (saves ~150-700ms per sample)
  - **Backward compatibility**: Gracefully handles v2 cache (recomputes embeddings if missing)

### November 5, 2025 - Reference Embedding Cache Fix
- **Cache validation bug fixed**: Reference embeddings now load correctly from cache
  - **Issue**: numpy saves dicts as 0-dim arrays, validation failed on `isinstance(ref_cache, dict)`
  - **Fix**: Added `.item()` extraction for numpy scalar arrays before validation
  - **Impact**: Saves 2-5 minutes per retrieval by reusing cached reference embeddings

### November 5, 2025 - Cross-Dataset Sample Caching (v2)
- **Sample metadata caching**: Cross-dataset retrieval optimization system
  - Caches visual/semantic metrics and EfficientNet predictions
  - Reusable across multiple dataset retrievals from the same source
  - **Cache location**: `cache_base_dir/sample_metadata_cache/{hash[:2]}/sample_{hash}_v{version}.json`
  - **Test coverage**: 16 comprehensive tests in `tests/test_sample_cache.py`

### November 19, 2025 - Caltech101 Torchvision Compatibility Fix
- **Class list mismatch fixed**: Aligned class_names with torchvision's actual implementation
  - **Issue**: Torchvision explicitly removes 'BACKGROUND_Google' category, leaving 101 classes (labels 0-100)
  - **Root cause**: MAVERIC had 102 classes including 'background_google', causing index mismatch
  - **Impact**: 'yin_yang' class showed 0 samples (index 101 doesn't exist in torchvision)
  - **Fix**: Updated class_names to match torchvision's 101 sorted categories (excluding BACKGROUND_Google)
  - **Result**: All 101 Caltech101 classes now correctly mapped to torchvision labels
  - **Location**: [elevater_datasets.py](maveric/datasets/elevater_datasets.py)

### November 18, 2025 - Caltech101 Dataset Fixes
- **Missing "leopards" class**: Added missing class to Caltech101 dataset
  - **Issue**: "leopards" class was completely missing from class names list
  - **Fix**: Added "leopards" at correct position in alphabetical ordering
  - **Location**: [elevater_datasets.py](maveric/datasets/elevater_datasets.py)

### November 13, 2025 - Statistics Display Improvements
- **Retrieval statistics fix**: Enhanced progress display for better clarity
  - Always shows cache hits and downloads (even if 0) for consistency
  - Improved verification: Processed = Cache Hits + Downloads
  - Better batch position tracking and index information display
  - **Location**: [progress.py](maveric/core/progress.py)

### November 2, 2025 - Critical Bug Fix
- **Class name extraction bug**: Fixed GTSRB showing only 3/43 classes due to underscore parsing issue
  - All datasets with underscores in class names now work correctly (e.g., `ahead_only`, `beware_of_ice_snow`)
  - Test script included: `test_class_name_extraction.py`

### October 30, 2025 - Major Performance & Reliability Updates

**Critical Configuration Changes**:
- `enable_target_class_quality`: **Default changed to `false`** (was `true`) - provides 50-70% faster retrieval

**New Features**:
- **Atomic file writes**: `save_json_atomic()` in `io_utils.py` prevents corruption on Google Drive/NFS
- **Enhanced cache validation**: Automatic detection and regeneration of corrupted cache files
- **Diagnostic logging**: Comprehensive file-based dataset debugging with directory structure analysis
- **Progress logging**: Long operations now show progress (CLIP loading, dataset loading, reference generation)

**Documentation**:
- New `docs/bugfixes/` directory with 88 KB of comprehensive bug fix documentation
- Complete retrieval analysis and performance optimization guides
- 10 complete CIFAR-100 experiment runs documented in `experiments/CIFAR100_Experiments.txt`

**Performance Impact**:
- 50-70% faster data retrieval (EfficientNet disabled by default)
- No more file corruption or hanging on network filesystems
- Better debugging capabilities with enhanced logging

See the "Recent Improvements" section below for detailed information.

## Architecture Overview

MAVERIC is a multi-modal dataset curation system for vision-language models. The codebase follows a modular architecture:

- **`maveric/main.py`**: Main MAVERIC class providing high-level API for retrieval, quality control, and model customization
- **`maveric/config.py`**: Dataclass-based configuration system (MAVERICConfig, TrainingConfig, ExperimentConfig)
- **`maveric/core/`**: Base interfaces, exceptions, abstract components, and progress tracking system
  - `base.py`: BaseComponent, BaseDataset, BaseMetric abstract classes
  - `interfaces.py`: RetrievalResult, QualityResult, CustomizationResult, ProgressCallback
  - `exceptions.py`: MAVERICError hierarchy (ConfigurationError, DatasetError, ModelError, CacheError)
  - `progress.py`: RealTimeStats for live statistics tracking
- **`maveric/retrieval/`**: Dataset retrieval and caching system with CLIP-based embedding
  - `retriever.py`: Main retrieval engine with quality metric computation
  - `cache_manager.py`: Smart caching for images, embeddings, and results
  - `sample_cache_manager.py`: **NEW** Cross-dataset sample metadata caching
  - `dataset_handlers.py`: Handlers for different dataset formats (REACT, etc.)
- **`maveric/quality/`**: Quality assessment metrics (visual, semantic, multimodal consistency)
  - `quality_controller.py`: Main quality control orchestration
  - `filters.py`: Quality-based filtering and dataset balancing
  - `metrics/`: Quality metric implementations organized by category
- **`maveric/customization/`**: Model fine-tuning with filtered data
  - `model_customizer.py`: High-level fine-tuning API
  - `training.py`: Training loop with Trainer and TrainingMonitor
  - `evaluation.py`: Model evaluation on test sets
- **`maveric/interactive/`**: Jupyter widgets for threshold selection and quality dashboards
  - `threshold_selector.py`: Interactive threshold tuning widget
  - `quality_dashboard.py`: Quality distribution visualization dashboard
  - `widgets.py`: Reusable UI components
- **`maveric/visualization/`**: Data distribution plots and sample galleries
  - `distributions.py`: MetricsVisualizer for distribution plots
  - `samples.py`: SampleVisualizer for image galleries
  - `interactive.py`: Full-featured MAVERICInteractiveQualityControl GUI
  - `plots.py`: Utility plotting functions
- **`maveric/datasets/`**: Unified ELEVATER benchmark dataset handler (official 20 datasets: 9 torchvision + 11 file-based)
  - `elevater_datasets.py`: ELEVATER benchmark dataset implementations
  - `dataset_factory.py`: Factory for creating dataset instances
- **`maveric/models/`**: CLIP model wrappers and factory patterns
  - `clip_wrapper.py`: CLIP model wrapper with utilities
  - `model_factory.py`: Factory for creating model instances
- **`maveric/utils/`**: Command-line interface, I/O utilities, logging, and visualization helpers
  - `cli.py`: Complete CLI for all MAVERIC operations
  - `io_utils.py`: File handling and data serialization
  - `logging.py`: Structured logging system
  - `visualization.py`: Visualization helper utilities

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
- Missing `sentence-transformers`: `pip install sentence-transformers` (required for target class quality metrics)
- Missing `scikit-learn`: `pip install scikit-learn` (required for cosine similarity calculations)
- Missing `langdetect`: `pip install langdetect` (required for text quality metrics)
- Missing `torchvision`: Required for EfficientNet-B0 in semantic quality assessment
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
# Retrieve samples (via experiment script for full control)
python experiments/01_data_retrieval.py --config config.yaml

# Retrieve samples WITHOUT EfficientNet (faster, skips Class_*_efficientNet_score)
python experiments/01_data_retrieval.py --config config.yaml --disable-efficientnet

# Apply quality control
maveric quality-control --input results.json --thresholds thresholds.json --output filtered.json

# Customize model
maveric customize --input filtered.json --model openai/clip-vit-base-patch32 --epochs 10 --output-dir ./models

# Visualize distributions
maveric visualize --input results.json --output-dir ./plots
```

## Experiment Scripts

The `experiments/` directory contains end-to-end workflows for different stages:

- **`00_setup.py`**: Environment setup script for automated installation and configuration
- **`01_data_retrieval.py`**: Data retrieval from source datasets with CLIP-based matching
- **`02_data_curation.py`**: Quality control and filtering of retrieved data
- **`03_model_customization.py`**: Model fine-tuning with curated datasets
- **`04_results_analysis.py`**: Analysis and visualization of experiment results
- **`maveric_config.yaml`**: Configuration file with optimal hyperparameters

Each script is designed to be run independently or as part of a complete pipeline.

## Configuration System

MAVERIC uses dataclass-based configuration in `config.py`:

- **MAVERICConfig**: Main system configuration (models, caching, quality thresholds, progress display)
- **TrainingConfig**: Model training parameters with regularization and augmentation
- **ExperimentConfig**: Experiment management and tracking

Key configuration options:
- `enable_real_time_stats`: Show live download/cache statistics during retrieval (default: true)
- `enable_target_class_quality`: Enable EfficientNet-based TargetClassQualityMetric (default: **false** for 50-70% faster retrieval, set to true for comprehensive quality assessment)
- `request_timeout`: HTTP request timeout in seconds (default: 5)
- `clip_model`: CLIP model to use (default: "ViT-B/32")
- `cache_base_dir`: Directory for caching downloaded images and results
- `batch_size`: Processing batch size
- `retrieval_rotation_size`: Samples per file when saving results and training data (default: 1000)
- `quality_metrics`: List of quality metrics to compute (default: visual: resolution, sharpness, color_diversity; semantic: text_quality, caption_length; multimodal: target_class_quality, multimodal_consistency)
- `metric_weights`: Weights for composite scoring across modalities (img2img: 0.4, txt2txt/img2txt/txt2img: 0.2 each)
- `seed`: Random seed for reproducible sampling (default: 42)

TrainingConfig key parameters:
- `epochs`: Number of training epochs (default: 10)
- `learning_rate`: Learning rate for optimizer (default: 1e-6)
- `weight_decay`: L2 regularization weight (default: 0.01)
- `use_regularization`: Enable MSE regularization to prevent catastrophic forgetting (default: true)
- `regularization_weight`: Weight for regularization loss (default: 0.5, range: 0.0-1.0)
- `use_augmentation`: Enable RandAugment data augmentation (default: true)
- `augmentation_strength`: RandAugment num_ops parameter (default: 2)
- `augmentation_magnitude`: RandAugment magnitude parameter (default: 9)
- `optimizer`: Optimizer type - adamw, adam, or sgd (default: "adamw")
- `scheduler`: Learning rate scheduler - cosine, linear, or constant (default: "cosine")
- `use_validation`: Enable validation split during training (default: true)
- `validation_method`: Validation strategy - stratified_kfold or simple_split (default: "stratified_kfold")

Configuration can be loaded from YAML/JSON files:
```python
config = MAVERICConfig.from_yaml('config.yaml')
maveric = MAVERIC.from_config_file('config.yaml')
```

## Key Components

### Quality Metrics (`maveric/quality/metrics/`)
**MAVERIC now properly implements all three metric categories:**

- **Visual metrics** (`visual_metrics.py`): Image-only quality assessment
  - `ResolutionMetric`: Image resolution evaluation  
  - `SharpnessMetric`: Laplacian variance-based sharpness
  - `ColorDiversityMetric`: Color channel standard deviation

- **Semantic metrics** (`semantic_metrics.py`): Text-only quality assessment ✅ **NOW ENABLED**
  - `TextQualityMetric`: Caption quality (length, vocabulary, language detection)
  - `CaptionLengthMetric`: Caption length appropriateness

- **Multimodal metrics** (`multimodal_metrics.py`): Cross-modal quality assessment
  - `MultimodalConsistencyMetric`: CLIP-based cross-modal alignment
  - `CrossModalAlignmentMetric`: Direct image-text similarity  
  - `TargetClassQualityMetric`: EfficientNet + CLIP for per-class quality assessment

### Core System (`maveric/core/`)
- **Base Classes**: Abstract components for datasets, metrics, and system components
- **Interfaces**: Result types and callback interfaces for consistent API
- **Exception Handling**: Centralized exception hierarchy for error management
- **Progress Tracking**: `RealTimeStats` system for live download/cache statistics display

### Retrieval System (`maveric/retrieval/`)
- CLIP-based embedding similarity matching with per-class quality assessment
- Smart caching system for images and embeddings
- Dataset handlers for different source formats
- **NEW**: Class mapping visualization showing target dataset classes → ImageNet-1K mappings
- **NEW**: Progress bar suppression for cleaner console output

### Interactive Dashboard (`maveric/interactive/`)
- **ThresholdSelector** (`threshold_selector.py`): Jupyter widget for real-time threshold tuning with metric weight controls
- **QualityDashboard** (`quality_dashboard.py`): Quality distribution visualization (updated to exclude global composite_quality, now uses imagenet_probability)
- **Widgets** (`widgets.py`): Reusable UI components including threshold sliders, weight controls, and metric selectors
- Sample gallery with filtering
- Interactive controls for metric weights with auto-normalizing sliders that maintain 1.0 sum

### Customization System (`maveric/customization/`)
- **ModelCustomizer** (`model_customizer.py`): High-level API for model fine-tuning with regularization
- **Trainer** (`training.py`): Training loop implementation with validation and monitoring
- **TrainingMonitor** (`training.py`): Real-time training metrics tracking and logging
- **Evaluator** (`evaluation.py`): Model evaluation on test sets with comprehensive metrics

### Visualization System (`maveric/visualization/`)
- **MetricsVisualizer** (`distributions.py`): Quality metric distribution plots and analysis
- **SampleVisualizer** (`samples.py`): Image galleries and sample inspection tools
- **Interactive GUI** (`interactive.py`): Full-featured interactive quality control interface for Jupyter/Colab
  - `MAVERICInteractiveQualityControl`: Main interactive controller class
  - `create_quality_control()`: Factory function for creating GUI instances
  - `start_interactive_gui()`: Convenience function for launching GUI
  - Automatic fallback when ipywidgets unavailable
- **Plotting Utilities** (`plots.py`): Class distribution, correlation matrices, quality comparisons

### Utilities (`maveric/utils/`)
- **CLI System** (`cli.py`): Complete command-line interface for all MAVERIC operations (retrieve, quality-control, customize, visualize)
- **I/O Utilities** (`io_utils.py`): File handling, data serialization, and configuration management
  - **NEW**: `save_json_atomic()`: Atomic write pattern for network filesystems (prevents corruption on Google Drive/NFS)
- **Logging** (`logging.py`): Structured logging system with configurable levels and formatters
- **Visualization Helpers** (`visualization.py`): Utility functions for plotting and data visualization

## Data Flow

1. **Retrieval**: Load source dataset → Generate CLIP embeddings → Match against target dataset embeddings
2. **Quality Assessment**: Apply visual/semantic metrics → Score each sample → Calculate composite quality scores per class
3. **Filtering**: Apply thresholds → Balance dataset → Export filtered results
4. **Customization**: Fine-tune model on filtered data → Evaluate performance

### Data Formats and Outputs

**Retrieval Results**: Saved as rotation files (JSON/pickle) with hierarchical structure:
- Image metadata (URL, caption, dimensions)
- Visual metrics: `resolution_score`, `sharpness_score`, `color_score`
- Semantic metrics: `text_quality_score`, `caption_length_score`
- Multimodal metrics: Per-class scores (e.g., `Class_airplane_img2img`, `Class_airplane_efficientNet_score`)
- Composite scores: `weighted_class_score`, `consistency` per class

**Curated Training Data**: Hierarchical directory structure to avoid NFS issues:
```
training_data/
├── class_000/
│   ├── batch_000/
│   │   ├── sample_000.jpg
│   │   ├── sample_001.jpg
│   │   └── ...
│   └── batch_001/
│       └── ...
└── class_001/
    └── ...
```

**Model Checkpoints**: PyTorch model files with training state:
- `best_model.pth`: Best model based on validation accuracy
- Training configuration and hyperparameters embedded
- Compatible with CLIP architecture for downstream tasks

### Advanced Quality Assessment

**Per-Class Target Quality Scoring**: **MAJOR UPDATE** - Target class quality is now calculated per-class instead of globally, following the same pattern as `hybrid_score` and `consistency`:
- For each target dataset class (e.g., CIFAR-10's 10 classes), a class-specific composite quality score is computed
- Uses EfficientNet-B0 + CLIP embeddings for semantic similarity with ImageNet classes
- Results in `Class_{class_name}_efficientNet_score` columns (e.g., `Class_airplane_efficientNet_score`) based on CLIP similarity with predicted ImageNet class
- Enables class-aware quality assessment - images are evaluated specifically for how well they represent each class
- Class selection combines similarity score with class-specific quality score using configurable weights

**Target Class Quality Metric**: **Properly categorized as multimodal** - Located in `multimodal_metrics.py` and used for per-class quality assessment:
- Uses EfficientNet-B0 (CPU-only) for universal image classification
- Employs CLIP for semantic similarity with ImageNet classes (more robust than sentence transformers)
- Pre-computes CLIP embeddings for ImageNet classes for efficiency
- Focuses on semantically relevant ImageNet classes based on caption content
- Works universally across all ELEVATER datasets without manual class mappings
- **Current implementation**: Returns CLIP similarity × ImageNet probability as final score
- **OPTIMIZED**: Batch processing computes mappings for all target classes using single EfficientNet inference
- Provides comprehensive quality scores considering both visual quality and semantic relevance

**Semantic Quality Filtering**: **NEW** - Pure text quality assessment now enabled by default:
- Text quality metrics filter poor captions (wrong language, too short/long, low vocabulary diversity)
- Caption length metrics ensure appropriate caption sizes
- Semantic filtering works alongside visual and multimodal quality assessment

**Class Selection Architecture**: Simplified class selection logic using weighted similarity scores:
- Weighted similarity scoring: Uses img2img, txt2txt, img2txt, txt2img metrics with configurable weights
- Pure similarity-based approach: Class selection based on weighted_class_score only
- Interactive controls: Real-time metric weight adjustment via Jupyter widgets

## Testing Strategy

- Unit tests for each component in `tests/`
- Configuration fixtures in `tests/conftest.py`
- Quality metrics validation with known datasets
- Integration tests for full pipeline workflows

## Cache Management

Images, embeddings, and reference data are cached in configurable directories:
- Image cache: JPEG compressed images for fast loading
- **Sample metadata cache**: **NEW** Cross-dataset sample caching for 60-85% speedup
- Embedding cache: Precomputed CLIP embeddings
- Results cache: Serialized retrieval and quality results
- Reference images cache: Reference images used for embedding generation (organized by dataset/class)
- Reference texts cache: Text templates and generated prompts for verification

### Cross-Dataset Sample Caching (v3 - UPDATED)

**Purpose**: Cache reusable data across multiple dataset retrievals to dramatically reduce processing time.

**What's Cached** (per sample - v3):
- Visual metrics (resolution, sharpness, color_diversity)
- Semantic metrics (text_quality, caption_length)
- **CLIP embeddings (image + text)** ⭐ NEW in v3 - base64 encoded
- EfficientNet predictions (ImageNet class + probability)

**What's NOT Cached**:
- Per-class similarity scores (`Class_{name}_img2img`, `Class_{name}_txt2txt`, etc.) - dataset-specific
- Class-specific quality scores - dataset-specific
- Dataset-specific reference comparisons

**Performance Impact**:
```
Cache v2 (without CLIP embeddings):
  First retrieval (CIFAR-10, 10k samples):    ~2.2 hours (builds cache)
  Second retrieval (CIFAR-100, 10k samples):  ~0.5 hours (75% faster)

Cache v3 (WITH CLIP embeddings):  ⭐ NEW
  First retrieval (CIFAR-10, 10k samples):    ~2.2 hours (builds cache)
  Second retrieval (CIFAR-100, 10k samples):  ~0.3 hours (85% faster!)
  Eliminates CLIP inference: saves 150-700ms per sample on cache hits

Total for 20 datasets: ~7.0 hrs vs 44.4 hrs (84% savings!)
```

**Storage**:
- **Per sample (v3)**: ~17KB (~500 bytes metrics + ~16KB embeddings)
- **270K samples**: ~4.5GB (trade-off: more storage for faster retrieval)
- **Per sample (v2)**: ~500 bytes (no embeddings)
- **270K samples**: ~135MB (v2 - less storage, slower retrieval)

**Configuration**:
```yaml
enable_sample_cache: true          # Enable/disable caching (default: true)
sample_cache_version: 3            # Cache format version (v3: includes CLIP embeddings)
```

**Version History**:
- **v3** (current): Caches CLIP embeddings + metrics + EfficientNet predictions
- **v2**: Caches metrics + EfficientNet predictions only (CLIP computed from cached images)
- **v1**: Initial implementation

**Cache Invalidation**:
- Increment `sample_cache_version` in config when metric computation changes
- Clear specific URL: `cache_manager.clear_cache(url="...")`
- Clear all: `cache_manager.clear_cache()`

### Cache Directory Structure
```
maveric_cache/
├── image_cache/                   # Cached downloaded images
│   └── {hash[:2]}/
│       └── img_{hash}.jpg
├── sample_metadata_cache/         # ⭐ Cross-dataset sample cache (v3: includes CLIP embeddings)
│   └── {hash[:2]}/
│       └── sample_{hash}_v3.json
├── reference_images/              # Reference samples per dataset
│   └── {dataset_name}/
│       └── {class_name}/
│           ├── ref_000.jpg
│           └── ...
├── reference_texts/               # Text templates per dataset
│   └── {dataset_name}_texts.json
└── embeddings/                    # Dataset-specific reference embeddings
    └── {dataset}_reference_embeddings.npz
```

### Sample Cache JSON Format (v3)
```json
{
  "cache_version": 3,
  "url": "https://...",
  "url_hash": "a1b2c3d4...",
  "text": "A photo of a cat",
  "last_updated": "2025-11-05T10:30:00Z",
  "visual_metrics": {
    "resolution_score": 0.895,
    "sharpness_score": 0.923,
    "color_score": 0.812
  },
  "semantic_metrics": {
    "text_quality_score": 0.850,
    "caption_length_score": 0.920
  },
  "clip_embeddings": {
    "image_embedding": "base64_encoded_string_of_numpy_array...",
    "text_embedding": "base64_encoded_string_of_numpy_array...",
    "image_shape": [1, 512],
    "text_shape": [1, 512],
    "dtype": "float32"
  },
  "efficientnet_predictions": {
    "imagenet_predicted_class": "tabby cat",
    "imagenet_probability": 0.892
  }
}
```

**Note**: v3 caches CLIP embeddings as base64-encoded numpy arrays. This increases file size (~17KB vs ~500 bytes) but eliminates CLIP inference on cache hits (saves 150-700ms per sample).

Reference texts files contain:
- `templates`: Original text templates used
- `class_names`: List of all class names in the dataset
- `generated_prompts`: Dictionary mapping each class to its generated prompts

## Dataset Support

MAVERIC supports all 20 official ELEVATER benchmark datasets through a unified handler:

### Torchvision-based Datasets (9):
- CIFAR-10, CIFAR-100
- Caltech101, Country211, EuroSAT
- Food101, GTSRB
- Oxford Flowers102, Oxford Pets

### File-based Datasets (11):
- DTD, FER2013, FGVCAircraft
- Hateful Memes, KITTI Distance, MNIST
- PatchCamelyon, RenderedSST2, RESISC45
- Stanford Cars, VOC2007

Torchvision datasets benefit from automatic downloading, standardized interfaces, and optimized loading.

### Caltech101 Special Notes

**Torchvision Behavior**: Torchvision's Caltech101 implementation automatically:
- Removes the 'BACKGROUND_Google' category
- Sorts remaining 101 categories alphabetically
- Assigns labels 0-100 to these categories

**Manual Download Workaround** (if torchvision download URLs are broken):
1. Download Caltech101 from Kaggle: https://www.kaggle.com/datasets/imbikramsaha/caltech-101
2. Place in: `{cache_dir}/datasets/caltech101/101_ObjectCategories/`
3. Torchvision will use the local files automatically
4. Note: You'll get 8,677 samples (torchvision excludes BACKGROUND_Google images)

**Expected Behavior**:
- Dataset size: 8,677 samples (not 9,144 - BACKGROUND_Google excluded)
- Number of classes: 101 (not 102)
- All classes including 'yin_yang' will have samples

### Important Notes for Large Datasets

**Food101 and Other Large Datasets**: Some torchvision datasets (particularly Food101, ~5GB with 75,750 training samples) require special handling:
- **First-time setup**: Automatic download via torchvision (one-time, ~5GB download)
- **Optimized class indexing**: Single-pass scan through dataset to build class-to-sample mapping
  - Original approach: O(n × c) = 101 classes × 75,750 samples = 7.6M iterations
  - Optimized approach: O(n) = 75,750 samples with progress updates
- **Progress logging**: Real-time progress updates during index building (20 checkpoints)
- **Google Drive considerations**: Initial scan may be slower on Google Drive vs. local storage

**What You'll See During Reference Sample Selection**:
```
Selecting FOOD101 sample data randomly...
  Dataset size: 75,750 samples
  Number of classes: 101
  Samples per class: 10
  Building class index map (one-time scan)...
    Progress: 3,787/75,750 (5.0%)
    Progress: 7,575/75,750 (10.0%)
    ...
    Progress: 75,750/75,750 (100.0%)
  ✅ Index map built. Processing classes...
  [1/101] Class 'apple_pie': 750 samples
  [2/101] Class 'baby_back_ribs': 750 samples
  ...
✅ Reference sampling complete: 101 classes, 1010 total images
```

**Performance Notes**:
- Index building takes ~2-5 minutes on Google Drive, ~30 seconds on local SSD
- Subsequent runs use cached embeddings (no need to rebuild)
- Other large datasets (Caltech101, Country211) benefit from same optimization

## Important Development Notes

### Package Structure
- Entry point: `setup.py` defines package metadata and dependencies
- Core package: `maveric/` contains all source code (~11,681 lines across 47 classes)
- Tests: `tests/` contains unit and integration tests
  - `test_optimization.py`: Validates EfficientNet batch processing optimizations
- Examples: `examples/` contains usage examples
  - `interactive_notebook.ipynb`: Interactive Jupyter notebook demonstrating MAVERIC features
- Experiments: `experiments/` contains end-to-end workflow scripts
  - `CIFAR100_Experiments.txt`: 10 complete experiment runs with manual hyperparameter tuning results (362 lines)
- Documentation: Comprehensive documentation suite
  - `README.md`: Main project documentation (16.6 KB)
  - `CLAUDE.md`: Developer guide for Claude Code (this file, 30+ KB)
  - `docs/bugfixes/`: Bug fix documentation suite (88 KB total, 8 files)
  - `docs/maveric-api-docs.md`: API reference documentation (12 KB)
  - `docs/detailed_documentation.txt`: Detailed API and architecture docs (16 KB)

### CLI Entry Point
The CLI entry point is correctly defined in `setup.py:49` as `maveric=maveric.utils.cli:main`, which points to the actual CLI implementation in `maveric/utils/cli.py`.

### Testing Environment Setup
- All tests automatically force CPU device via `conftest.py` device fixture
- Random seeds are set for reproducibility (numpy=42, torch=42)
- No pytest configuration files - uses defaults
- For headless environments (Docker/CI): set `MPLBACKEND=Agg` before running tests

### Development Dependencies Structure
The project uses layered requirements:
- `requirements.txt`: Core runtime dependencies only
- `requirements-dev.txt`: Includes base requirements via `-r requirements.txt` plus dev tools

### Configuration Architecture Details
Configuration uses dataclasses in `config.py` with three main classes:
- `MAVERICConfig`: System config with intelligent defaults (auto device detection, directory creation)
- `TrainingConfig`: Model training parameters
- `ExperimentConfig`: Experiment management and tracking

Key config features:
- YAML/JSON loading support
- Auto device detection when set to "auto"
- Automatic directory creation in `__post_init__`
- Smart path handling: Replaces inaccessible `/content/` paths with local alternatives when not in Colab
- Legacy field mapping: Automatically maps deprecated config fields to current versions

### Google Colab and Drive Compatibility
MAVERIC is optimized for Google Colab environments:
- **Setup script**: `experiments/00_setup.py` handles Colab-specific setup including Drive mounting
- **Path handling**: Config system detects and replaces inaccessible Colab paths
- **Hierarchical storage**: Avoids Google Drive NFS mount issues with hierarchical file organization
- **Progress tracking**: Configurable progress displays suitable for Colab notebooks
- **Environment variables**: Automatic setup of `MAVERIC_BASE_DIR`, `MAVERIC_CACHE_DIR`, etc.
- **System dependencies**: Automated installation of required system packages

## Performance & Architecture Improvements

### CPU-Only Data Retrieval
- **TargetClassQualityMetric** now uses EfficientNet-B0 on CPU during data retrieval
- Eliminates GPU memory usage during the data collection phase
- Maintains high-quality assessment while reducing hardware requirements

### Disabling EfficientNet for Faster Retrieval
**NEW**: You can now disable EfficientNet-based quality metrics for significantly faster data retrieval:

**Command-line flag**:
```bash
python experiments/01_data_retrieval.py --config config.yaml --disable-efficientnet
```

**Configuration file**:
```yaml
enable_target_class_quality: false  # Disable EfficientNet calculations
```

**What gets skipped when disabled**:
- `Class_{class_name}_efficientNet_score` fields are not computed
- `Class_{class_name}_clip_similarity_to_imagenet` fields are not computed
- `imagenet_predicted_class` and `imagenet_probability` fields are not added
- EfficientNet-B0 model loading and inference are skipped entirely

**Performance impact**:
- **~50-70% faster** data retrieval (depending on dataset and hardware)
- Significantly reduced CPU usage during retrieval
- All other quality metrics (visual, semantic, similarity-based) remain available

**When to disable**:
- Initial data exploration when you want quick results
- When EfficientNet scores are not needed for your filtering criteria
- Limited computational resources or time constraints
- Working with very large datasets (>100k samples)

**When to keep enabled**:
- Need per-class ImageNet-based quality assessment
- Filtering based on EfficientNet scores
- Final production data curation with comprehensive metrics

### Data Curation Compatibility

The `02_data_curation.py` script **automatically handles both types of data**:

**With EfficientNet metrics** (standard retrieval):
- All quality thresholds are applied, including EfficientNet-based ones
- Full range of filtering options available

**Without EfficientNet metrics** (`--disable-efficientnet` retrieval):
- Script automatically detects missing EfficientNet fields
- Filters skip missing metrics gracefully (no errors)
- All other thresholds (visual, semantic, similarity) are still applied
- Quality control works identically, just with fewer metrics

**Example workflow**:
```bash
# Step 1: Fast retrieval without EfficientNet
python experiments/01_data_retrieval.py --config config.yaml --disable-efficientnet

# Step 2: Curation works automatically (no special flags needed)
python experiments/02_data_curation.py --input-dir results/cifar10/raw --dataset-name cifar10 --config config.yaml
```

The curation script will display:
```
ℹ️  EfficientNet metrics not present (data retrieved with --disable-efficientnet)
   Visual, semantic, and similarity metrics are still available for filtering
```

**Note**: If your quality thresholds include EfficientNet-based metrics (like `imagenet_probability`), they will be automatically skipped without causing errors.

### Optimized Quality Score Calculation
- **Batch EfficientNet Processing**: EfficientNet inference runs only once per image, not once per target class
- **Probability Reuse**: Same ImageNet probabilities are reused for all target class mappings
- **Performance Improvement**: Reduces computational overhead from O(N) to O(1) EfficientNet calls per image (where N = number of target classes)
- **Memory Efficiency**: Computes all ImageNet mappings from a single probability tensor

### Model Customization with Regularization
- **Locked-Text Tuning**: Only vision encoder is fine-tuned, text encoder remains frozen
- **MSE Regularization**: Prevents catastrophic forgetting by maintaining similarity to original vision weights
- **Configurable Regularization Weight**: Control trade-off between adaptation and preservation (default: 0.5)
- **Formula**: `total_loss = task_loss + regularization_weight × MSE(current_weights, original_weights)`
- **Performance**: Optimal `regularization_weight` typically in range [0.4, 0.6] based on empirical results

### Progress Bar Management  
- **Real-time statistics**: Set `enable_real_time_stats: false` in config to disable live download/cache statistics
- **Console-friendly**: Configurable progress display for production environments

### Per-Class Quality Architecture
- **Eliminated global quality scores**: No more single `composite_quality` per sample - now using per-class `imagenet_probability`
- **Class-specific assessment**: Quality evaluated relative to each target class using CLIP-ImageNet mappings
- **Consistent data structure**: Follows same pattern as similarity metrics (`Class_{name}_{metric}`)
- **Enhanced class selection**: Combines similarity and quality at the class level with configurable weighting

### Memory Optimization
- **Efficient caching**: Smart image and embedding cache management
- **Rotation files**: Large datasets automatically split into manageable chunks (default: 1000 samples per file)
- **Hierarchical file structure**: Implements hierarchical organization to avoid Google Drive NFS mount errors
- **Resource management**: Better GPU/CPU resource allocation during different phases

### Recent Improvements (Latest Commits)

**November 18, 2025 - Caltech101 Dataset Fixes**:
- **Missing "leopards" class**: Added complete missing class to Caltech101 (now correctly has 102 classes)
- **Class list formatting**: Fixed trailing comma and proper alphabetical ordering
- **Impact**: Full Caltech101 dataset support with all official classes

**November 13, 2025 - Statistics Display Improvements**:
- **Enhanced progress tracking**: Improved retrieval statistics display for better clarity
- **Consistent reporting**: Always shows cache hits and downloads (even if 0)
- **Verification formula**: Processed = Cache Hits + Downloads
- **Location**: [progress.py](maveric/core/progress.py)

**November 2, 2025 - Critical Bug Fix**:
- **Class Name Extraction Bug Fix**: Fixed critical bug in `interactive.py` where class names containing underscores (e.g., GTSRB's `ahead_only`, `beware_of_ice_snow`) were incorrectly parsed
  - **Impact**: GTSRB dataset only showed 3/43 classes (pedestrians, stop, yield) - the only classes without underscores
  - **Root Cause**: Used `split('_')[1]` which broke on class names with underscores
  - **Fix**: Proper suffix removal logic that handles arbitrary underscores in class names
  - **Affected**: Lines 275 and 1142 in `visualization/interactive.py`
  - **Tested**: All GTSRB class names now extracted correctly including complex names like `no_passing_for_vehicles_over_3_5_metric_tons`

**October 30, 2025 - Major Performance & Reliability Updates**:
- **EfficientNet Default Changed**: Now disabled by default (`enable_target_class_quality: false`) for 50-70% faster data retrieval
- **Atomic File Writes**: New `save_json_atomic()` function prevents file corruption and hanging on network filesystems (Google Drive/NFS)
- **Enhanced Cache Validation**: Automatic detection and regeneration of corrupted cache files with clear warning messages
- **Diagnostic Logging**: Comprehensive logging for file-based dataset issues, including directory structure analysis and per-class loading status
- **Progress Logging for Long Operations**: Shows progress during CLIP model loading, dataset loading, and reference generation (eliminates "frozen" appearance)
- **File-Based Dataset Bug Fix** (commit cc0b48f): Fixed "no images found" errors due to incorrect directory structure assumptions
- **Retrieval Performance Improvements** (commit 7e17c73): Multiple optimizations in retrieval module for faster processing

**Previous Improvements**:
- **Optional EfficientNet**: EfficientNet calculations can be disabled via `enable_target_class_quality: false` for ~50-70% faster retrieval (commit 8d54ac5)
- **Hierarchical file structure**: Avoids Google Drive NFS mount issues by organizing data hierarchically (commit 101170c)
- **Image copying optimization**: Pre-copies images during curation for faster validation during customization (commit 97aa1bd)
- **Enhanced progress tracking**: Improved progress bars with better timeout handling for data saving (commits 9468f77, f908a6c)
- **Debug logging**: Added detailed logs for slow validation processes (commit d961256)
- **URL tracking**: Outputs file URLs when downloads fail during curation for debugging (commit 3459d12)
- **Cleaner output**: Save data output cleaning for better console readability (commit cf27497)
- **Interactive GUI enhancements**:
  - Reset button for threshold controls
  - Random sample display on each "Show Samples" click
  - Combobox for quality threshold presets
  - EfficientNet prediction visualization tab
  - Class distribution in EfficientNet filtering

### Bug Fixes Documentation

**NEW**: Comprehensive bug fix documentation available in `docs/bugfixes/` directory:

- **`BUGFIX_SUMMARY.md`**: Complete summary of critical bug fixes implemented on October 30, 2025
  - EfficientNet default change (true → false)
  - Atomic file writes for network filesystems
  - Enhanced cache validation with corruption detection
  - Progress logging for long-running operations
  - Diagnostic logging for file-based datasets

- **`DIAGNOSTIC_LOGGING_IMPROVEMENT.md`**: Detailed guide for enhanced diagnostic logging
  - Directory structure validation
  - Per-class loading status tracking
  - Empty directory warnings
  - Comprehensive failure reports

- **`RETRIEVAL_ANALYSIS.md` series**: Complete technical analysis (23 KB total)
  - Architecture documentation with code locations
  - Performance regression analysis
  - Optimization recommendations
  - Index guide for navigating the analysis

- **`CHANGELOG_BUGFIXES.md`**: Concise changelog format for quick reference

**Performance Impact**: These bug fixes collectively provide:
- 50-70% faster data retrieval (EfficientNet disabled by default)
- Eliminated file corruption on Google Drive/NFS (atomic writes)
- Better debugging capabilities (diagnostic logging)

See [docs/bugfixes/README.md](docs/bugfixes/README.md) for detailed documentation.