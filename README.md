# MAVERIC: Multi-modal Adaptive Visual Embedding Retrieval with Integrated Consistency

MAVERIC is a sophisticated quality control system for multi-modal dataset curation, designed to improve the performance of vision-language models through intelligent data filtering and selection.

## Features

- **Multi-modal Quality Assessment**: Comprehensive quality metrics including visual (sharpness, resolution, color diversity) and semantic (consistency, alignment) measures
- **Interactive Threshold Selection**: Real-time GUI for finding optimal quality thresholds
- **Efficient Caching**: Smart image caching system for handling large-scale datasets
- **Model Customization**: Fine-tune vision-language models with high-quality curated data
- **Visualization Tools**: Rich visualization of quality distributions and sample galleries
- **Extensible Architecture**: Easy to add new quality metrics, datasets, and models

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

## Installation

### Standard Installation
```bash
pip install maveric
```

### Development Installation
```bash
git clone https://github.com/avbarbaros/maveric.git
cd maveric

# Install dependencies first
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### Docker/Headless Environment Setup
For environments without display (Docker, CI/CD, remote servers):

```bash
# Install with headless OpenCV
pip install opencv-python-headless

# Set matplotlib backend for headless environments
export MPLBACKEND=Agg
```

### Common Installation Issues

1. **Missing CLIP**: Install with `pip install openai-clip`
2. **OpenGL errors**: Use `opencv-python-headless` instead of `opencv-python`
3. **Matplotlib style errors**: Set `MPLBACKEND=Agg` for headless environments
4. **PyTorch CPU-only**: Install with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## Quick Start

```python
from maveric import MAVERIC, MAVERICConfig

# Initialize MAVERIC
config = MAVERICConfig(
    cache_base_dir="/content/drive/MyDrive/MAVERIC/maveric_cache",
    clip_model="ViT-B/32"
)
maveric = MAVERIC(config)

# Retrieve and filter data
retrieval_result = maveric.retrieve(
    dataset_name="react-vl/react-retrieval-datasets",
    target_dataset="cifar100",
    num_samples=100000
)

# Launch interactive dashboard
dashboard = maveric.launch_dashboard(retrieval_result)

# Apply quality control
quality_result = maveric.quality_control(
    retrieval_result,
    thresholds={'sharpness_score': 0.85, 'consistency': 0.80}
)

# Customize model
customization_result = maveric.customize_model(
    quality_result,
    model_name="openai/clip-vit-base-patch32"
)
```

## Testing

### Basic Testing Commands

```bash
# Run all tests
pytest

# Run with coverage (requires: pip install pytest-cov)
pytest --cov=maveric --cov-report=html

# Run specific test file
pytest tests/test_quality_metrics.py

# Run specific test
pytest tests/test_main.py::TestMAVERIC::test_retrieve

# Run tests with verbose output
pytest -v

# Run tests with short traceback for cleaner output
pytest --tb=short
```

**Note**: Coverage commands require `pytest-cov`. Install with:
```bash
pip install pytest-cov
# OR install dev dependencies
pip install -e ".[dev]"
```

### Headless Environment Testing

**Important**: For Docker containers, CI/CD pipelines, or remote servers without display capabilities, use these commands:

```bash
# Basic headless testing (RECOMMENDED)
MPLBACKEND=Agg pytest

# Headless with coverage (requires pytest-cov)
MPLBACKEND=Agg pytest --cov=maveric --cov-report=html

# Headless with verbose output (no extra dependencies)
MPLBACKEND=Agg pytest -v --tb=short

# Headless minimal (fastest, least output)
MPLBACKEND=Agg pytest -q

# Set environment variable permanently (Linux/Mac)
export MPLBACKEND=Agg
pytest

# Set environment variable permanently (Windows)
set MPLBACKEND=Agg
pytest
```

**Why `MPLBACKEND=Agg`?** 
- **Prevents display errors**: Matplotlib tries to open GUI windows by default, which fails in headless environments
- **Enables visualization tests**: All plotting and visualization tests require this backend in Docker/CI environments  
- **Universal compatibility**: Works across local development, containers, and cloud environments
- **No functionality loss**: Agg backend supports all matplotlib features except interactive displays

### Docker Testing

```bash
# Test inside Docker container
docker run -it --rm -v $(pwd):/workspace -w /workspace python:3.9 bash -c "
  pip install -r requirements.txt && 
  pip install -e .[dev] && 
  MPLBACKEND=Agg pytest
"

# Using docker-compose (if available)
docker-compose run --rm test bash -c "MPLBACKEND=Agg pytest"
```

### CI/CD Pipeline Commands

```bash
# Basic CI testing (no extra dependencies)
MPLBACKEND=Agg pytest --junitxml=test-results.xml

# With coverage for CI (requires pytest-cov)
MPLBACKEND=Agg pytest --junitxml=test-results.xml --cov=maveric --cov-report=xml

# With parallel testing (requires pytest-xdist)  
MPLBACKEND=Agg pytest -n auto --dist=loadfile

# Memory-efficient testing for constrained environments
MPLBACKEND=Agg pytest --maxfail=1 --tb=short -q

# Complete CI setup with dependencies
pip install pytest pytest-cov pytest-xdist
MPLBACKEND=Agg pytest --junitxml=test-results.xml --cov=maveric --cov-report=xml
```

## CLI Usage

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

## ELEVATER Experiments on Google Colab

To evaluate MAVERIC's quality-driven filtering effectiveness across all ELEVATER datasets on Google Colab T4 GPU:

### Prerequisites
- Google Colab account with T4 GPU runtime
- Google Drive with 10+ GB free space
- Stable internet connection
- 3-4 hours of computation time

### Overview

These experiments systematically test MAVERIC's ability to improve dataset quality through intelligent filtering across 20 diverse computer vision datasets from the ELEVATER benchmark.

### ELEVATER Datasets Processed

The experiments process these 20 datasets:
- Cars, CIFAR10, CIFAR100, DTD, EuroSAT, FER2013
- FGVCAircraft, Flowers102, Food101, GTSRB, HatefulMemes
- MNIST, OxfordIIITPet, PCAM, RESISC45, RenderedSST2
- StanfordCars, STL10, SUN397, SVHN

### Experiment Scripts

The `experiments/` folder contains sequentially numbered scripts for running comprehensive ELEVATER dataset experiments:

#### 1. Setup and Installation (`01_colab_setup.py`)
**Purpose**: Install MAVERIC and dependencies on Google Colab T4 environment
**Usage**:
```python
python /content/maveric/experiments/01_colab_setup.py
```
**Expected Behavior**:
- Checks GPU availability and system information
- Installs system dependencies for headless environment
- Sets up environment variables (`MPLBACKEND=Agg`)
- Clones MAVERIC repository from GitHub
- Installs dependencies from `requirements.txt`
- Installs MAVERIC in development mode
- Tests installation with import checks
- **Duration**: ~10-15 minutes

#### 2. Google Drive Cache Setup (`02_google_drive_setup.py`)
**Purpose**: Configure Google Drive integration for caching and results storage
**Usage**:
```python
python /content/maveric/experiments/02_google_drive_setup.py
```
**Expected Behavior**:
- Mounts Google Drive to `/content/drive`
- Loads configuration from `maveric_config.yaml`
- Creates cache directory structure on Google Drive
- Sets up environment variables for cache paths
- Tests read/write access to cache directories
- Shows disk usage and space availability
- Creates experiment log template with dataset checklist
- **Duration**: ~2-3 minutes

#### 3. ELEVATER Datasets Experiments (`03_elevater_experiments.py`)
**Purpose**: Run MAVERIC quality filtering on all 20 ELEVATER datasets
**Usage**:
```python
python /content/maveric/experiments/03_elevater_experiments.py
```
**Expected Behavior**:
- Processes datasets: Cars, CIFAR10/100, DTD, EuroSAT, FER2013, FGVCAircraft, Flowers102, Food101, GTSRB, HatefulMemes, MNIST, OxfordIIITPet, PCAM, RESISC45, RenderedSST2, StanfordCars, STL10, SUN397, SVHN
- For each dataset:
  - Retrieves samples using CLIP embedding similarity
  - Applies quality assessment metrics (visual, semantic, multimodal)
  - Filters data based on configured thresholds
  - Saves detailed results and summaries
  - Generates visualizations (quality distributions, sample galleries)
  - Updates experiment progress log
- Creates comprehensive experiment summary with aggregate statistics
- **Duration**: ~2-3 hours (5-10 minutes per dataset)
- **Output**: Individual results per dataset + overall summary

#### 4. Results Analysis and Visualization (`04_results_analysis.py`)
**Purpose**: Analyze experiment results and generate comprehensive reports
**Usage**:
```python
python /content/maveric/experiments/04_results_analysis.py
```
**Expected Behavior**:
- Loads experiment results from all datasets
- Creates pandas DataFrame for statistical analysis
- Generates summary statistics (retention rates, quality scores, execution times)
- Creates comprehensive visualizations:
  - Success rate pie chart
  - Retention rate distribution histogram
  - Sample counts by dataset
  - Execution time analysis
  - Quality scores heatmap
  - Detailed retention rate comparison
- Generates markdown analysis report with:
  - Executive summary with key findings
  - Dataset-specific performance results
  - Quality metrics analysis
  - Conclusions and recommendations
- **Duration**: ~5-10 minutes

### Configuration

The experiments use `experiments/maveric_config.yaml` for configuration:
- **Quality Thresholds**: Adjustable quality filtering thresholds for different metrics
- **Processing Settings**: Batch sizes, sample limits, caching options optimized for T4 GPU
- **ELEVATER Datasets**: Complete list of 20 datasets to process
- **Output Settings**: Results format, visualization options, experiment tracking

### Expected Results Structure

After running all experiments, you'll find in `/content/drive/MyDrive/MAVERIC/maveric_experiments/`:
```
maveric_experiments/
├── maveric_config.yaml                 # Configuration backup
├── experiment_log.md                   # Progress tracking with checkboxes
├── elevater_experiment_summary.json    # Overall experiment results
├── maveric_analysis_report.md          # Comprehensive analysis report
├── maveric.log                         # Detailed execution logs
├── elevater_results/                   # Individual dataset results
│   ├── Cars/
│   ├── CIFAR10/
│   └── ... (20 datasets)
└── analysis_visualizations/            # Charts and plots
    ├── experiment_overview.png
    └── retention_rates_detailed.png
```

### Key Research Insights

The experiments will quantify MAVERIC's effectiveness by measuring:
- **Retention Rates**: Percentage of data passing quality filters per dataset
- **Quality Score Distributions**: How quality metrics vary across datasets
- **Filtering Effectiveness**: Data reduction vs. quality improvement trade-offs
- **Processing Efficiency**: Execution times and resource utilization
- **Cross-Dataset Patterns**: Which datasets benefit most from quality filtering

### Configuration Options

Edit `experiments/maveric_config.yaml` to adjust:
- **Quality Thresholds**: Adjustable quality filtering thresholds for different metrics
- **Processing Settings**: Batch sizes, sample limits, caching options optimized for T4 GPU
- **ELEVATER Datasets**: Complete list of 20 datasets to process
- **Output Settings**: Results format, visualization options, experiment tracking

### Troubleshooting

**Common Issues:**
- GPU not detected: Ensure T4 runtime is selected in Colab
- Google Drive mounting fails: Re-authenticate Drive permissions
- Out of memory errors: Reduce batch_size in config
- Long execution times: Check internet connection stability

**Performance Tips:**
- Use GPU runtime for faster processing
- Keep Colab tab active to prevent timeouts
- Monitor Google Drive storage space
- Run during off-peak hours for better performance

### Running the Complete Pipeline

Execute scripts in sequence:
```bash
# Step 1: Setup (run once)
python /content/maveric/experiments/01_colab_setup.py

# Step 2: Configure cache (run once) 
python /content/maveric/experiments/02_google_drive_setup.py

# Step 3: Run experiments (main analysis)
python /content/maveric/experiments/03_elevater_experiments.py

# Step 4: Generate analysis report
python /content/maveric/experiments/04_results_analysis.py
```

## Documentation

Full documentation is available at [https://maveric.readthedocs.io](https://maveric.readthedocs.io)

## Citation

If you use MAVERIC in your research, please cite:

```bibtex
@software{maveric2025,
  title={MAVERIC: Multi-modal Adaptive Visual Embedding Retrieval with Integrated Consistency},
  author={Ali V. Barbaros},
  year={2025},
  url={https://github.com/avbarbaros/maveric}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.