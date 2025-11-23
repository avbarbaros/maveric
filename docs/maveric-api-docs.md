# MAVERIC API Reference

## Overview

MAVERIC (Multi-modal Adaptive Visual Embedding Retrieval with Integrated Consistency) is a sophisticated quality control system for multi-modal dataset curation. This document provides a comprehensive API reference for all MAVERIC components.

## Table of Contents

1. [Main API](#main-api)
2. [Configuration](#configuration)
3. [Retrieval](#retrieval)
4. [Quality Control](#quality-control)
5. [Model Customization](#model-customization)
6. [Visualization](#visualization)
7. [Interactive Components](#interactive-components)
8. [Utilities](#utilities)

## Main API

### MAVERIC

The main entry point for the MAVERIC library.

```python
from maveric import MAVERIC, MAVERICConfig

# Initialize with default config
maveric = MAVERIC()

# Initialize with custom config
config = MAVERICConfig(cache_base_dir="/content/drive/MyDrive/MAVERIC/maveric_cache")
maveric = MAVERIC(config)

# Load from config file
maveric = MAVERIC.from_config_file("config.yaml")
```

#### Methods

##### retrieve()
```python
retrieval_result = maveric.retrieve(
    dataset_name="react-vl/react-retrieval-datasets",
    target_dataset="cifar100",
    num_samples=100000,
    start_index=0,
    cache_results=True
)
```

**Parameters:**
- `dataset_name` (str): Source dataset name
- `target_dataset` (str): Target dataset name (e.g., 'cifar10', 'cifar100')
- `num_samples` (int, optional): Number of samples to retrieve
- `start_index` (int): Starting index for retrieval
- `cache_results` (bool): Whether to cache results

**Returns:** `RetrievalResult` object

##### quality_control()
```python
quality_result = maveric.quality_control(
    data=retrieval_result,
    thresholds={'sharpness_score': 0.85, 'consistency': 0.80},
    balance_strategy='median'
)
```

**Parameters:**
- `data`: RetrievalResult, DataFrame, or file path
- `thresholds` (dict, optional): Quality thresholds
- `weights` (dict, optional): Metric weights
- `balance_strategy` (str): 'none', 'median', 'mean', 'min', 'max'

**Returns:** `QualityResult` object

##### customize_model()
```python
customization_result = maveric.customize_model(
    quality_result=quality_result,
    model_name="openai/clip-vit-base-patch32",
    training_config=TrainingConfig(epochs=10)
)
```

**Parameters:**
- `quality_result`: Filtered data from quality_control()
- `model_name` (str, optional): Base model to customize
- `training_config` (TrainingConfig, optional): Training configuration

**Returns:** `CustomizationResult` object

##### Interactive GUI
For interactive quality control in Jupyter notebooks, use the visualization package:

```python
from maveric.visualization import start_interactive_gui
gui = start_interactive_gui('cifar100', config_file=None)
```

This launches a full-featured interactive dashboard with threshold adjustment, sample galleries, and quality visualization.

## Configuration

### MAVERICConfig

Main configuration class for MAVERIC.

```python
from maveric import MAVERICConfig

config = MAVERICConfig(
    # Model settings
    clip_model="ViT-B/32",
    device="auto",
    
    # Cache settings
    cache_base_dir="/content/drive/MyDrive/MAVERIC/maveric_cache",
    enable_image_cache=True,
    cache_format="jpg",
    
    # Quality thresholds
    default_thresholds={
        'weighted_class_score': 0.493,
        'consistency': 0.796,
        'resolution_score': 0.370,
        'sharpness_score': 0.880,
        'color_score': 0.768
    }
)
```

#### Key Parameters

- `clip_model`: CLIP model variant ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')
- `device`: Computing device ('auto', 'cuda', 'cpu')
- `cache_base_dir`: Base directory for all caches
- `n_reference_images`: Number of reference images per class
- `retrieval_rotation_size`: Samples per file when saving
- `quality_metrics`: List of quality metrics to compute
- `metric_weights`: Weights for multi-modal scoring

### TrainingConfig

Configuration for model customization.

```python
from maveric import TrainingConfig

training_config = TrainingConfig(
    epochs=10,
    learning_rate=1e-5,
    use_regularization=True,
    regularization_weight=0.5,
    use_augmentation=True,
    optimizer="adamw",
    scheduler="cosine"
)
```

## Retrieval

### Retriever

Handles retrieval and scoring of samples.

```python
from maveric.retrieval import Retriever, CacheManager

cache_manager = CacheManager("/path/to/cache")
retriever = Retriever(
    clip_model="ViT-B/32",
    device="cuda",
    cache_manager=cache_manager
)
```

### RetrievalResult

Result container for retrieval operations.

```python
# Access retrieval results
print(f"Total samples: {retrieval_result.total_samples}")
print(f"Class distribution: {retrieval_result.class_distribution}")

# Convert to DataFrame
df = retrieval_result.to_dataframe()

# Save results
retrieval_result.save("results.json")

# Load results
loaded_result = RetrievalResult.load("results.json")
```

## Quality Control

### QualityController

Main controller for quality assessment and filtering.

```python
from maveric.quality import QualityController

qc = QualityController(data)

# Set thresholds
qc.set_threshold('sharpness_score', 0.85)
qc.set_threshold('consistency', 0.80)

# Apply filtering
filtered_count = qc.apply_thresholds()

# Balance dataset
balanced_data = qc.balance_dataset(
    strategy='median',
    min_samples=15,
    max_samples=200
)
```

### Quality Metrics

#### Visual Metrics

```python
from maveric.quality.metrics import (
    ResolutionMetric,
    SharpnessMetric,
    ColorDiversityMetric,
    FeatureRichnessMetric
)

# Create metric instance
sharpness = SharpnessMetric()

# Compute score for an image
score = sharpness.compute(image, metadata)
```

#### Multimodal Metrics

```python
from maveric.quality.metrics import (
    MultimodalConsistencyMetric,
    CrossModalAlignmentMetric
)

# Consistency metric requires reference embeddings
consistency = MultimodalConsistencyMetric(
    clip_model="ViT-B/32",
    reference_embeddings=ref_embeddings,
    text_embeddings=text_embeddings
)
```

### Filters

```python
from maveric.quality.filters import (
    ThresholdFilter,
    BalancedFilter,
    PercentileFilter
)

# Create threshold filter
threshold_filter = ThresholdFilter({
    'sharpness_score': 0.85,
    'consistency': 0.80
})

# Apply filter
filtered_df = threshold_filter.apply(data_df)
```

## Model Customization

### ModelCustomizer

Handles model fine-tuning with filtered data.

```python
from maveric.customization import ModelCustomizer

customizer = ModelCustomizer(
    base_model_name="openai/clip-vit-base-patch32",
    device="cuda"
)

result = customizer.customize(
    quality_result=quality_result,
    training_config=training_config,
    target_dataset_name="cifar100",
    class_names=class_names
)
```

### CustomizationResult

```python
print(f"Test accuracy: {result.test_accuracy:.2f}%")
print(f"Improvement: {result.improvement:+.2f}%")
print(f"Per-class accuracies: {result.class_accuracies}")

# Access training history
plt.plot(result.training_history['train_loss'])
plt.plot(result.training_history['val_loss'])
```

## Visualization

### MetricsVisualizer

```python
from maveric.visualization import MetricsVisualizer

viz = MetricsVisualizer(style='seaborn')

# Plot single metric distribution
fig = viz.plot_metric_distribution(
    data_df,
    'sharpness_score',
    threshold=0.85
)

# Plot multiple metrics
fig = viz.plot_multi_metric_distributions(
    data_df,
    ['sharpness_score', 'consistency', 'color_score'],
    thresholds={'sharpness_score': 0.85}
)
```

### SampleVisualizer

```python
from maveric.visualization import SampleVisualizer

sample_viz = SampleVisualizer()

# Visualize samples
fig = sample_viz.visualize_samples(
    data_df,
    n_samples=10,
    sample_type='diverse'  # 'random', 'best', 'worst', 'diverse'
)

# Create quality grid
fig = sample_viz.create_quality_grid(
    data_df,
    metric='weighted_class_score',
    grid_size=(3, 3)
)
```

## Interactive Components

### MAVERICInteractiveQualityControl

Full-featured interactive GUI for Jupyter/Colab notebooks:

```python
from maveric.visualization import start_interactive_gui

# Launch interactive quality control
gui = start_interactive_gui(
    dataset_name='cifar100',
    config_file=None  # Or path to config YAML
)

# Display the GUI
gui
```

The interactive GUI provides:
- Real-time threshold adjustment with live filtering
- Quality metric distribution visualization
- Sample gallery with filtering capabilities
- Class distribution analysis
- Dataset balancing controls
- EfficientNet prediction visualization (if available)

## Utilities

### Logging

```python
from maveric.utils import setup_logging, get_logger

# Setup logging
setup_logging(
    level="INFO",
    log_to_file=True,
    log_file="maveric.log"
)

# Get logger
logger = get_logger("my_module")
logger.info("Processing started")
```

### I/O Utilities

```python
from maveric.utils import load_json, save_json, download_file

# Load/save JSON
data = load_json("data.json")
save_json(data, "output.json")

# Download file
download_file(
    url="https://example.com/image.jpg",
    output_path="downloaded_image.jpg"
)
```

## Dataset Support

### Built-in Datasets

- CIFAR-10: `'cifar10'`
- CIFAR-100: `'cifar100'`
- ELEVATER datasets: Various vision benchmarks

### Adding Custom Datasets

```python
from maveric.datasets import DatasetFactory, BaseDataset

class MyDataset(BaseDataset):
    @property
    def name(self):
        return "my_dataset"
    
    @property
    def class_names(self):
        return ["class1", "class2", "class3"]
    
    def get_reference_samples(self, n_per_class):
        # Return dict mapping class names to image lists
        pass
    
    def get_text_templates(self):
        return ["a photo of a {}", "an image of a {}"]

# Register dataset
DatasetFactory.register("my_dataset", MyDataset)
```

## Common Workflows

### Complete Pipeline

```python
# 1. Initialize
maveric = MAVERIC(MAVERICConfig(cache_base_dir="/content/drive/MyDrive/MAVERIC/maveric_cache"))

# 2. Retrieve
retrieval_result = maveric.retrieve(
    "react-vl/react-retrieval-datasets",
    "cifar100",
    num_samples=100000
)

# 3. Quality Control
quality_result = maveric.quality_control(
    retrieval_result,
    thresholds={'sharpness_score': 0.85},
    balance_strategy='median'
)

# 4. Customize Model
customization_result = maveric.customize_model(
    quality_result,
    training_config=TrainingConfig(epochs=10)
)

# 5. Evaluate
print(f"Improvement: {customization_result.improvement:+.2f}%")
```

### Interactive Threshold Selection

```python
from maveric.visualization import start_interactive_gui

# Launch interactive GUI
gui = start_interactive_gui('cifar100', config_file=None)

# After finding optimal thresholds in GUI
optimal_config = {
    'thresholds': {
        'sharpness_score': 0.87,
        'consistency': 0.82
    }
}

# Apply optimized thresholds
quality_result = maveric.quality_control(
    retrieval_result,
    **optimal_config
)
```

## Error Handling

MAVERIC uses custom exceptions for different error types:

```python
from maveric.core.exceptions import (
    MAVERICError,        # Base exception
    ConfigurationError,  # Configuration issues
    DatasetError,       # Dataset loading/processing
    ModelError,         # Model loading/inference
    CacheError          # Cache operations
)

try:
    result = maveric.retrieve(dataset_name, target)
except DatasetError as e:
    print(f"Dataset error: {e}")
except MAVERICError as e:
    print(f"MAVERIC error: {e}")
```

## Performance Tips

1. **Enable Caching**: Always use caching for large-scale retrieval
   ```python
   config = MAVERICConfig(enable_image_cache=True)
   ```

2. **Batch Processing**: Process in batches for memory efficiency
   ```python
   config = MAVERICConfig(retrieval_rotation_size=1000)
   ```

3. **GPU Usage**: Use GPU for faster processing
   ```python
   config = MAVERICConfig(device="cuda")
   ```

4. **Parallel Processing**: For CPU-bound operations
   ```python
   config = MAVERICConfig(num_workers=4)  # For data loading
   ```

## Support for ELEVATER Benchmark

MAVERIC supports all ELEVATER benchmark datasets:

```python
# List available ELEVATER datasets
from maveric.datasets.elevater_datasets import ELEVATERDataset
print(ELEVATERDataset.ELEVATER_DATASETS.keys())

# Use ELEVATER dataset
retrieval_result = maveric.retrieve(
    dataset_name="your-source-dataset",
    target_dataset="oxford_pets"  # ELEVATER dataset
)
```