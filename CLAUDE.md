# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Install hyperparameter search dependencies
pip install -e ".[dev]"  # Includes numpy, scikit-learn for search utilities
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

### Hyperparameter Search
MAVERIC includes a systematic hyperparameter search tool for optimizing model performance:

```bash
# Focused search around optimal regularization (recommended)
python experiments/05_hyperparameter_search.py \
    --input data/training/ \
    --config maveric_config.yaml \
    --output results/hp_search/ \
    --search-type focused

# Fine-grained regularization weight search
python experiments/05_hyperparameter_search.py \
    --input data/training/ \
    --config maveric_config.yaml \
    --output results/hp_search/ \
    --search-type regularization

# Learning rate optimization
python experiments/05_hyperparameter_search.py \
    --input data/training/ \
    --config maveric_config.yaml \
    --output results/hp_search/ \
    --search-type learning_rate

# Random search (faster exploration)
python experiments/05_hyperparameter_search.py \
    --input data/training/ \
    --config maveric_config.yaml \
    --output results/hp_search/ \
    --search-type broad \
    --method random \
    -n 30
```

See `experiments/HYPERPARAMETER_SEARCH.md` for detailed guide and search strategies.

## Experiment Scripts

The `experiments/` directory contains end-to-end workflows for different stages:

- **`00_setup.py`**: Environment setup script for automated installation and configuration
- **`01_data_retrieval.py`**: Data retrieval from source datasets with CLIP-based matching
- **`02_data_curation.py`**: Quality control and filtering of retrieved data
- **`03_model_customization.py`**: Model fine-tuning with curated datasets
- **`04_results_analysis.py`**: Analysis and visualization of experiment results
- **`05_hyperparameter_search.py`**: Systematic hyperparameter optimization
- **`maveric_config.yaml`**: Configuration file with optimal hyperparameters
- **`run_hp_search.sh`**: Shell script for running hyperparameter searches
- **`HYPERPARAMETER_SEARCH.md`**: Comprehensive guide for hyperparameter optimization

Each script is designed to be run independently or as part of a complete pipeline.

## Configuration System

MAVERIC uses dataclass-based configuration in `config.py`:

- **MAVERICConfig**: Main system configuration (models, caching, quality thresholds, progress display)
- **TrainingConfig**: Model training parameters with regularization and augmentation
- **ExperimentConfig**: Experiment management and tracking

Key configuration options:
- `enable_real_time_stats`: Show live download/cache statistics during retrieval (default: true)
- `enable_target_class_quality`: Enable EfficientNet-based TargetClassQualityMetric (default: true, set to false for faster retrieval)
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
- Embedding cache: Precomputed CLIP embeddings
- Results cache: Serialized retrieval and quality results
- Reference images cache: Reference images used for embedding generation (organized by dataset/class)
- Reference texts cache: Text templates and generated prompts for verification

### Reference Data Storage Structure
```
maveric_cache/
├── reference_images/
│   └── {dataset_name}/
│       └── {class_name}/
│           ├── ref_000.jpg
│           ├── ref_001.jpg
│           └── ...
└── reference_texts/
    └── {dataset_name}_texts.json
```

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
- Core package: `maveric/` contains all source code
- Tests: `tests/` contains unit and integration tests
- Examples: `examples/` contains usage examples
- Experiments: `experiments/` contains end-to-end workflow scripts
- Documentation: `README.md`, `CLAUDE.md`, `docs/`, and `experiments/HYPERPARAMETER_SEARCH.md`

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
- **Optional EfficientNet**: EfficientNet calculations can be disabled via `enable_target_class_quality: false` for ~50-70% faster retrieval (commit 8d54ac5)
- **Hierarchical file structure**: Avoids Google Drive NFS mount issues by organizing data hierarchically (commit 101170c)
- **Image copying optimization**: Pre-copies images during curation for faster validation during customization (commit 97aa1bd)
- **Enhanced progress tracking**: Improved progress bars with better timeout handling for data saving (commits 9468f77, f908a6c)
- **Debug logging**: Added detailed logs for slow validation processes (commit d961256)
- **Timeout configuration**: Configurable timeouts for HTTP requests (default: 5s, configurable up to 10s+) (commit 50b0b1a)
- **URL tracking**: Outputs file URLs when downloads fail during curation for debugging (commit 3459d12)
- **Cleaner output**: Save data output cleaning for better console readability (commit cf27497)
- **Interactive GUI enhancements**:
  - Reset button for threshold controls
  - Random sample display on each "Show Samples" click
  - Combobox for quality threshold presets
  - EfficientNet prediction visualization tab
  - Class distribution in EfficientNet filtering