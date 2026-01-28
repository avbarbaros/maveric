# File-Based Datasets - Quick Start

## TL;DR

File-based datasets (FER2013, PCAM, RESISC45, etc.) **don't auto-download** like torchvision datasets. You must **manually download and organize them** before running `03_model_customization.py`.

---

## The Problem

```bash
# This works ✅ (torchvision datasets auto-download)
python experiments/03_model_customization.py \
    --input ./results/cifar10/curated/ \
    --config experiments/maveric_config.yaml

# This fails ❌ (file-based datasets need manual setup)
python experiments/03_model_customization.py \
    --input ./results/fer2013/curated/ \
    --config experiments/maveric_config.yaml
# Error: Failed to load test data for fer2013
```

**Why?** Torchvision datasets have built-in download support. File-based datasets don't.

---

## Quick Solutions

### Option 1: Manual Download (Recommended)

**Required Directory Structure**:
```
data/elevater/{dataset_name}/
├── train/              # Training images
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/               # Test images (REQUIRED for evaluation)
    ├── class_1/
    ├── class_2/
    └── ...
```

**Fastest to Set Up: FER2013**

1. Download: https://www.kaggle.com/datasets/msambare/fer2013
2. Extract to `data/elevater/fer2013/`
3. Done! Dataset already has train/test splits organized by class

```bash
# Download (requires Kaggle CLI)
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d fer2013_raw

# Organize
mkdir -p data/elevater/fer2013/train
mkdir -p data/elevater/fer2013/test
cp -r fer2013_raw/train/* data/elevater/fer2013/train/
cp -r fer2013_raw/test/* data/elevater/fer2013/test/

# Test
python -c "from maveric.datasets import get_dataset; \
    d = get_dataset('fer2013', train=False, root='./data'); \
    print(f'Test samples: {len(d._dataset) if d._dataset else 0}')"
```

---

### Option 2: Use Torchvision Datasets Instead

**Workaround**: Train on file-based data, evaluate on torchvision datasets

```bash
# Retrieve from file-based dataset (no test data needed)
python experiments/01_data_retrieval.py \
    --dataset fer2013 \
    --config experiments/maveric_config.yaml

# Curate (no test data needed)
python experiments/02_data_curation.py \
    --input-dir ./results/fer2013/raw \
    --dataset-name fer2013 \
    --config experiments/maveric_config.yaml

# Train on FER2013 data, but evaluate on CIFAR-10 instead
python experiments/03_model_customization.py \
    --input ./results/fer2013/curated/ \
    --config experiments/maveric_config.yaml \
    --target-dataset cifar10  # ← Use CIFAR-10 for evaluation
```

This allows you to use file-based data for training without setting up test data.

---

## Which Datasets Are Affected?

### Torchvision Datasets ✅ (Auto-Download Works)
- CIFAR-10, CIFAR-100
- MNIST
- Caltech101, Country211, DTD, EuroSAT
- Food101, GTSRB, FGVCAircraft
- Oxford Flowers102, Oxford Pets

### File-Based Datasets ❌ (Manual Setup Required)
1. **FER2013** - Facial expressions (7 classes) - [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. **PCAM** - Lymph node classification (2 classes) - [GitHub](https://github.com/basveeling/pcam)
3. **RenderedSST2** - Sentiment analysis (2 classes) - [ELEVATER](https://github.com/Computer-Vision-in-the-Wild/ELEVATER)
4. **RESISC45** - Remote sensing (45 classes) - [OneDrive](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)
5. **StanfordCars** - Car recognition (196 classes) - [Stanford](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
6. **VOC2007** - Object recognition (20 classes) - [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
7. **HatefulMemes** - Hate speech (2 classes) - [Facebook](https://hatefulmemeschallenge.com/)
8. **Kitti Distance** - Car distance (4 classes) - [KITTI](http://www.cvlibs.net/datasets/kitti/)

---

## Expected Directory Paths

When running `03_model_customization.py`, MAVERIC looks for test data at:

```
{cache_base_dir}/{dataset_name}/datasets/elevater/{dataset_name}/test/
```

**Example for FER2013**:
```
# If cache_base_dir = "./cache" and dataset = "fer2013"
./cache/fer2013/datasets/elevater/fer2013/test/angry/001.jpg
./cache/fer2013/datasets/elevater/fer2013/test/happy/002.jpg
...
```

**Default if no cache_base_dir configured**:
```
./data/elevater/{dataset_name}/test/
```

---

## Verification

Test if your setup is correct:

```python
from maveric.datasets import get_dataset

# Test FER2013 (or any file-based dataset)
dataset = get_dataset('fer2013', train=False, root='./data')

if dataset._dataset is None:
    print("❌ Test data not found - check directory structure")
else:
    print(f"✅ Test data loaded: {len(dataset._dataset)} samples")
```

---

## Full Documentation

See [FILE_BASED_DATASETS_GUIDE.md](FILE_BASED_DATASETS_GUIDE.md) for:
- Complete download links for all 8 datasets
- Dataset-specific extraction scripts
- Expected class names for each dataset
- Detailed setup instructions

---

## Recommended Next Steps

1. **Start with FER2013** (easiest - already organized by Kaggle uploader)
2. If FER2013 works, expand to others as needed
3. Or use **Option 2 workaround** to train on file-based data without test setup

**Questions?** Check the detailed guide: [FILE_BASED_DATASETS_GUIDE.md](FILE_BASED_DATASETS_GUIDE.md)
