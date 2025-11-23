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

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y $(cat system-requirements.txt | grep -v "^#" | xargs)

# Install Python dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### System Requirements
MAVERIC requires several system packages for computer vision and ML operations. These are listed in `system-requirements.txt`:

- `libgl1-mesa-glx`: OpenGL support for OpenCV image operations
- `libglib2.0-0`: GLib library for low-level system operations
- `libsm6`: X11 Session Management library for GUI applications
- `libxext6`: X11 Extension library for display operations  
- `libxrender-dev`: X11 Render extension for graphics rendering
- `libgomp1`: GNU OpenMP runtime for parallel processing

### Automated Setup Script
For Google Colab or automated environments, use the provided setup script:

```bash
cd experiments
python 01_setup.py --config maveric_config.yaml
```

This script will:
- Install system dependencies from `system-requirements.txt`
- Mount Google Drive (if in Colab)
- Set up environment variables including `MAVERIC_BASE_DIR`
- Create necessary directories based on configuration
- Install MAVERIC package and dependencies
- Validate the installation

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

# Initialize MAVERIC with base directory configuration
config = MAVERICConfig(
    maveric_base_dir="/content/drive/MyDrive/MAVERIC",  # Base directory for all MAVERIC files
    cache_base_dir="/content/drive/MyDrive/MAVERIC/maveric_cache",
    results_dir="/content/drive/MyDrive/MAVERIC/maveric_experiments", 
    clip_model="ViT-B/32"
)
maveric = MAVERIC(config)

# Retrieve and filter data
retrieval_result = maveric.retrieve(
    dataset_name="react-vl/react-retrieval-datasets",
    target_dataset="cifar100",
    num_samples=100000
)

# Launch interactive dashboard (for Jupyter/Colab)
from maveric.visualization import start_interactive_gui
gui = start_interactive_gui('cifar100', config_file=None)

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

## Configuration

MAVERIC uses YAML-based configuration for easy management. The configuration system supports hierarchical directory structure through the `maveric_base_dir` parameter.

### Configuration File Structure

```yaml
# Base configuration - all paths will be relative to this
maveric_base_dir: "/content/drive/MyDrive/MAVERIC"
cache_base_dir: "/content/drive/MyDrive/MAVERIC/maveric_cache"
results_dir: "/content/drive/MyDrive/MAVERIC/maveric_experiments"

# Model settings
clip_model: "ViT-B/32"
batch_size: 64
device: "cuda"

# Quality thresholds
quality_thresholds:
  sharpness_score: 0.7
  consistency: 0.7
  clip_similarity: 0.6
```

### Environment Variables

The setup script automatically configures these environment variables:

- `MAVERIC_BASE_DIR`: Base directory for all MAVERIC files
- `MAVERIC_CACHE_DIR`: Directory for caching images and embeddings  
- `MAVERIC_RESULTS_DIR`: Directory for experiment results and logs
- `MAVERIC_CONFIG_PATH`: Path to the configuration file
- `HF_HOME`: Hugging Face model cache directory

### Loading Configuration

```python
from maveric import MAVERIC

# Load from YAML file
maveric = MAVERIC.from_config_file('experiments/maveric_config.yaml')

# Or use environment variable
import os
config_path = os.getenv('MAVERIC_CONFIG_PATH', 'maveric_config.yaml')
maveric = MAVERIC.from_config_file(config_path)
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

#### 1. Setup and Installation (`01_setup.py`)
**Purpose**: Complete MAVERIC setup including installation and Google Drive integration
**Usage**:
```python
python /content/maveric/experiments/01_setup.py --config /path/to/maveric_config.yaml
```
**Command Line Options**:
- `--config`, `-c`: Path to MAVERIC configuration YAML file (required)
- `--help`, `-h`: Show help message and usage examples

**Example Usage**:
```bash
# Using relative path
python 01_setup.py --config ./maveric_config.yaml

# Using absolute path 
python 01_setup.py --config /content/drive/MyDrive/MAVERIC/config.yaml

# Short form
python 01_setup.py -c maveric_config.yaml
```

**Expected Behavior**:
- Validates configuration file exists and is valid YAML
- Checks GPU availability and system information
- Installs system dependencies for headless environment
- Mounts Google Drive to `/content/drive`
- Sets up environment variables from config (`MPLBACKEND=Agg`, cache paths)
- Creates cache directory structure on Google Drive
- Clones MAVERIC repository from GitHub (if needed)
- Installs dependencies from `requirements.txt`
- Installs MAVERIC in development mode
- Tests installation with import checks
- Tests read/write access to cache directories
- Backs up configuration to Google Drive
- Shows setup summary with disk usage and configuration details
- **Duration**: ~10-15 minutes

#### 2. ELEVATER Datasets Experiments (`03_elevater_experiments.py`)
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

#### 3. Results Analysis and Visualization (`04_results_analysis.py`)
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
# Step 1: Setup and configuration (run once)
python /content/maveric/experiments/01_setup.py --config /content/maveric/experiments/maveric_config.yaml

# Step 2: Run experiments (main analysis)
python /content/maveric/experiments/03_elevater_experiments.py

# Step 3: Generate analysis report
python /content/maveric/experiments/04_results_analysis.py
```

**Alternative setup options**:
```bash
# Using relative path (if running from experiments directory)
cd /content/maveric/experiments
python 01_setup.py --config ./maveric_config.yaml

# Using custom config file
python 01_setup.py --config /path/to/custom_config.yaml

# Show help and usage examples
python 01_setup.py --help
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