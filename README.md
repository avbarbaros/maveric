# MAVERIC: Multi-modal Adaptive Visual Embedding Retrieval with Integrated Consistency

MAVERIC is a sophisticated quality control system for multi-modal dataset curation, designed to improve the performance of vision-language models through intelligent data filtering and selection.

## Features

- **Multi-modal Quality Assessment**: Comprehensive quality metrics including visual (sharpness, resolution, color diversity) and semantic (consistency, alignment) measures
- **Interactive Threshold Selection**: Real-time GUI for finding optimal quality thresholds
- **Efficient Caching**: Smart image caching system for handling large-scale datasets
- **Model Customization**: Fine-tune vision-language models with high-quality curated data
- **Visualization Tools**: Rich visualization of quality distributions and sample galleries
- **Extensible Architecture**: Easy to add new quality metrics, datasets, and models

## Installation

```bash
pip install maveric
```

For development:
```bash
git clone https://github.com/avbarbaros/maveric.git
cd maveric
pip install -e ".[dev]"
```

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

---

# .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
cache/
data/
outputs/
*.log
*.json
*.pkl
*.pth
*.pt

# Except configuration examples
!examples/*.json
!configs/*.yaml