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
    cache_base_dir="/path/to/cache",
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