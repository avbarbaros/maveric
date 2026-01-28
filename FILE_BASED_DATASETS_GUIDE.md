# File-Based Datasets - Test Data Setup Guide

## Problem Summary

When running `03_model_customization.py`, MAVERIC successfully evaluates **torchvision datasets** (CIFAR-10, CIFAR-100, MNIST, etc.) but **fails to load test data for file-based datasets** (FER2013, HatefulMemes, PCAM, etc.).

**Root Cause**: File-based datasets don't have automatic download support like torchvision datasets. The `_load_dataset()` method does nothing for file-based datasets (`pass`), leaving `self._dataset = None`, which causes test data loading to fail during evaluation.

---

## File-Based Datasets in MAVERIC

The following 8 datasets are marked as `'type': 'file_based'`:

1. **FER2013** - Facial Expression Recognition (7 classes)
2. **HatefulMemes** - Hate speech detection (2 classes)
3. **Kitti Distance** - Car distance estimation (4 classes)
4. **PatchCamelyon (PCAM)** - Lymph node classification (2 classes)
5. **RenderedSST2** - Sentiment analysis (2 classes)
6. **RESISC45** - Remote sensing (45 classes)
7. **StanfordCars** - Car recognition (196 classes)
8. **VOC2007** - Object recognition (20 classes)

---

## Solution Overview

There are **three approaches** to solve this issue:

### **Option 1: Manual Download + Directory Structure** (Recommended for now)
- Download datasets manually from official sources
- Organize into expected directory structure
- Works immediately without code changes

### **Option 2: Implement PyTorch Dataset Wrappers** (Best long-term solution)
- Create custom PyTorch Dataset classes for each file-based dataset
- Similar to how torchvision datasets work
- Requires code implementation

### **Option 3: Skip Evaluation for File-Based Datasets** (Workaround)
- Use file-based datasets for retrieval only (not evaluation)
- Train on file-based data, evaluate on torchvision datasets
- Not ideal but allows workflow continuation

---

## Option 1: Manual Download + Directory Structure

### Expected Directory Structure

MAVERIC expects file-based datasets to be organized as follows:

```
{root}/elevater/{dataset_name}/
├── train/              # Training split (for reference samples)
│   ├── class_1/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── class_2/
│   │   └── ...
│   └── ...
└── test/               # Test split (for evaluation)
    ├── class_1/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── class_2/
    │   └── ...
    └── ...
```

**Important Notes**:
- `{root}` is the value passed to `get_dataset()` (default: `./data`)
- For model customization, root is typically: `{cache_base_dir}/{dataset_name}/datasets`
- Example full path: `./cache/fer2013/datasets/elevater/fer2013/test/angry/001.jpg`
- Supported image formats: `.jpg`, `.png`, `.jpeg`
- Class folder names should match ELEVATER class names (see below for each dataset)

### Dataset-Specific Download Instructions

#### 1. FER2013 (Facial Expression Recognition)

**Download Source**: [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

**Class Names** (7 classes):
```
angry, disgusted, fearful, happy, neutral, sad, surprised
```

**Directory Structure**:
```
data/elevater/fer2013/
├── train/
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    ├── angry/
    ├── disgusted/
    ├── fearful/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprised/
```

**Setup Steps**:
```bash
# 1. Download from Kaggle (requires Kaggle account)
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d fer2013_raw

# 2. Organize into MAVERIC structure
mkdir -p data/elevater/fer2013/train
mkdir -p data/elevater/fer2013/test

# The Kaggle dataset already has train/test splits in subdirectories
cp -r fer2013_raw/train/* data/elevater/fer2013/train/
cp -r fer2013_raw/test/* data/elevater/fer2013/test/
```

---

#### 2. PCAM (PatchCamelyon)

**Download Source**: [PCAM GitHub](https://github.com/basveeling/pcam)

**Class Names** (2 classes):
```
lymph_node
lymph_node_containing_metastatic_tumor_tissue
```

**Note**: Class names with spaces should use underscores in folder names (MAVERIC's `sanitize_filename()` handles this)

**Directory Structure**:
```
data/elevater/patchcamelyon/
├── train/
│   ├── lymph_node/
│   └── lymph_node_containing_metastatic_tumor_tissue/
└── test/
    ├── lymph_node/
    └── lymph_node_containing_metastatic_tumor_tissue/
```

**Setup Steps**:
```bash
# 1. Download PCAM dataset
# Follow instructions at https://github.com/basveeling/pcam
# You'll get .h5 files with images and labels

# 2. Extract images and organize
# (This requires Python script to read .h5 files and save as images)
python extract_pcam.py  # See script example below

# 3. Move to MAVERIC structure
mkdir -p data/elevater/patchcamelyon/train
mkdir -p data/elevater/patchcamelyon/test
# ... (organize extracted images)
```

**Example extraction script** (`extract_pcam.py`):
```python
import h5py
import numpy as np
from PIL import Image
from pathlib import Path

def extract_pcam(h5_path, output_dir, split='train'):
    """Extract PCAM images from .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        images = f['x'][:]
        labels = f['y'][:]

    # Create class directories
    class_0_dir = Path(output_dir) / split / 'lymph_node'
    class_1_dir = Path(output_dir) / split / 'lymph_node_containing_metastatic_tumor_tissue'
    class_0_dir.mkdir(parents=True, exist_ok=True)
    class_1_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    for i, (img, label) in enumerate(zip(images, labels)):
        target_dir = class_1_dir if label[0] == 1 else class_0_dir
        img_pil = Image.fromarray(img)
        img_pil.save(target_dir / f'{i:06d}.png')

    print(f"Extracted {len(images)} images to {output_dir}/{split}")

# Usage
extract_pcam('camelyonpatch_level_2_split_train_x.h5', 'data/elevater/patchcamelyon', 'train')
extract_pcam('camelyonpatch_level_2_split_test_x.h5', 'data/elevater/patchcamelyon', 'test')
```

---

#### 3. RenderedSST2 (Sentiment Analysis)

**Download Source**: [ELEVATER GitHub](https://github.com/Computer-Vision-in-the-Wild/ELEVATER)

**Class Names** (2 classes):
```
negative
positive
```

**Directory Structure**:
```
data/elevater/rendered_sst2/
├── train/
│   ├── negative/
│   └── positive/
└── test/
    ├── negative/
    └── positive/
```

**Setup Steps**:
```bash
# 1. Clone ELEVATER repository
git clone https://github.com/Computer-Vision-in-the-Wild/ELEVATER.git

# 2. Download RenderedSST2 (follow ELEVATER instructions)
cd ELEVATER
# ... (follow their download scripts)

# 3. Organize into MAVERIC structure
mkdir -p data/elevater/rendered_sst2/train
mkdir -p data/elevater/rendered_sst2/test
# ... (copy images to class folders)
```

---

#### 4. RESISC45 (Remote Sensing)

**Download Source**: [OneDrive Link](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)

**Class Names** (45 classes):
```
airplane, airport, baseball_diamond, basketball_court, beach, bridge, chaparral,
church, circular_farmland, cloud, commercial_area, dense_residential,
desert, forest, freeway, golf_course, ground_track_field, harbor,
industrial_area, intersection, island, lake, meadow, medium_residential,
mobile_home_park, mountain, overpass, palace, parking_lot, railway,
railway_station, rectangular_farmland, river, roundabout, runway,
sea_ice, ship, snowberg, sparse_residential, stadium, storage_tank,
tennis_court, terrace, thermal_power_station, wetland
```

**Directory Structure**:
```
data/elevater/resisc45/
├── train/
│   ├── airplane/
│   ├── airport/
│   └── ... (45 classes total)
└── test/
    ├── airplane/
    ├── airport/
    └── ... (45 classes total)
```

**Setup Steps**:
```bash
# 1. Download RESISC45 from OneDrive (requires manual download)
# Extract the .rar file

# 2. Split into train/test (RESISC45 doesn't have official split)
# Use 80/20 split or create your own
python split_resisc45.py

# 3. Organize into MAVERIC structure
mkdir -p data/elevater/resisc45/train
mkdir -p data/elevater/resisc45/test
# ... (move split data)
```

**Example split script** (`split_resisc45.py`):
```python
from pathlib import Path
import shutil
import random

def split_resisc45(source_dir, output_dir, train_ratio=0.8):
    """Split RESISC45 into train/test sets."""
    source = Path(source_dir)
    output = Path(output_dir)

    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = list(class_dir.glob('*.jpg'))
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Create directories
        train_dir = output / 'train' / class_name
        test_dir = output / 'test' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy(img, train_dir / img.name)
        for img in test_images:
            shutil.copy(img, test_dir / img.name)

        print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

# Usage
split_resisc45('NWPU-RESISC45', 'data/elevater/resisc45')
```

---

#### 5. StanfordCars

**Download Source**: [Stanford Cars Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

**Class Names**: 196 car models (see `elevater_datasets.py` lines 1009-1204 for complete list)

**Directory Structure**:
```
data/elevater/stanford_cars/
├── train/
│   ├── Acura_Integra_Type_R_2001/
│   ├── Acura_RL_Sedan_2012/
│   └── ... (196 classes)
└── test/
    ├── Acura_Integra_Type_R_2001/
    ├── Acura_RL_Sedan_2012/
    └── ... (196 classes)
```

**Note**: Class names with spaces are converted to underscores (e.g., "Acura Integra Type R 2001" → "Acura_Integra_Type_R_2001")

**Setup Steps**:
```bash
# 1. Download Stanford Cars dataset
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget http://ai.stanford.edu/~jkrause/car196/car_devkit.tgz

# 2. Extract archives
tar -xzf cars_train.tgz
tar -xzf cars_test.tgz
tar -xzf car_devkit.tgz

# 3. Organize using metadata (requires Python script)
python organize_stanford_cars.py

# 4. Move to MAVERIC structure
mkdir -p data/elevater/stanford_cars
mv organized/train data/elevater/stanford_cars/
mv organized/test data/elevater/stanford_cars/
```

---

#### 6. VOC2007

**Download Source**: [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)

**Class Names** (20 classes):
```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor
```

**Directory Structure**:
```
data/elevater/voc2007/
├── train/
│   ├── aeroplane/
│   ├── bicycle/
│   └── ... (20 classes)
└── test/
    ├── aeroplane/
    ├── bicycle/
    └── ... (20 classes)
```

**Setup Steps**:
```bash
# 1. Download VOC2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# 2. Extract
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

# 3. Organize by class (VOC2007 uses XML annotations)
python organize_voc2007.py

# 4. Move to MAVERIC structure
mkdir -p data/elevater/voc2007
mv organized/train data/elevater/voc2007/
mv organized/test data/elevater/voc2007/
```

**Example organization script** (`organize_voc2007.py`):
```python
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def organize_voc2007(voc_root, output_dir):
    """Organize VOC2007 images by class."""
    voc = Path(voc_root) / 'VOCdevkit' / 'VOC2007'
    output = Path(output_dir)

    for split in ['train', 'test']:
        # Read image set file
        imageset_file = voc / 'ImageSets' / 'Main' / f'{split}.txt'
        with open(imageset_file) as f:
            image_ids = [line.strip() for line in f]

        # Process each image
        for img_id in image_ids:
            # Read annotation
            ann_file = voc / 'Annotations' / f'{img_id}.xml'
            tree = ET.parse(ann_file)
            root = tree.getroot()

            # Get all object classes in this image
            classes = set()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes.add(class_name)

            # Copy image to each class directory
            img_file = voc / 'JPEGImages' / f'{img_id}.jpg'
            for class_name in classes:
                class_dir = output / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_file, class_dir / f'{img_id}.jpg')

        print(f"Organized {split} split: {len(image_ids)} images")

# Usage
organize_voc2007('.', 'data/elevater/voc2007')
```

---

#### 7. HatefulMemes

**Download Source**: [Facebook Research](https://hatefulmemeschallenge.com/)

**Class Names** (2 classes):
```
meme
hatespeech_meme
```

**Directory Structure**:
```
data/elevater/hateful_memes/
├── train/
│   ├── meme/
│   └── hatespeech_meme/
└── test/
    ├── meme/
    └── hatespeech_meme/
```

**Setup Steps**: Requires registration and download from Facebook Research

---

#### 8. Kitti Distance

**Download Source**: [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/)

**Class Names** (4 classes):
```
a_photo_i_took_of_a_car_on_my_left_or_right_side
a_photo_i_took_with_a_car_nearby
a_photo_i_took_with_a_car_far_away
a_photo_i_took_with_no_car
```

**Directory Structure**:
```
data/elevater/kitti_distance/
├── train/
│   ├── a_photo_i_took_of_a_car_on_my_left_or_right_side/
│   ├── a_photo_i_took_with_a_car_nearby/
│   ├── a_photo_i_took_with_a_car_far_away/
│   └── a_photo_i_took_with_no_car/
└── test/
    └── ... (same 4 classes)
```

**Setup Steps**: Requires downloading KITTI dataset and manually organizing by distance labels

---

## Option 2: Implement PyTorch Dataset Wrappers (Future Enhancement)

This is the **best long-term solution** but requires significant code implementation.

### Implementation Plan

1. **Create dataset wrapper classes** in `elevater_datasets.py`:
   ```python
   class FER2013Dataset(torch.utils.data.Dataset):
       def __init__(self, root, train=True, download=False):
           # Download logic
           # Load from CSV
           # Index images
   ```

2. **Add to torchvision loaders** in `_load_torchvision_dataset()`:
   ```python
   dataset_loaders = {
       # ... existing loaders ...
       'fer2013': lambda: FER2013Dataset(
           root=self.root, train=self.train, download=self.download),
   }
   ```

3. **Change dataset type** in `ELEVATER_DATASETS`:
   ```python
   'fer2013': {
       'num_classes': 7,
       'task': 'classification',
       'type': 'torchvision',  # Changed from 'file_based'
       'class_names': [...]
   }
   ```

### Benefits
- Automatic download support
- Consistent interface with torchvision datasets
- Better integration with MAVERIC pipeline

### Challenges
- Each dataset requires custom download/parsing logic
- Some datasets require registration (HatefulMemes, Kitti)
- Dataset-specific preprocessing may be needed

---

## Option 3: Workaround - Skip Evaluation

If you only need file-based datasets for **data retrieval** (not evaluation), you can:

1. **Use file-based datasets for retrieval only**:
   ```bash
   python experiments/01_data_retrieval.py --dataset fer2013 --config config.yaml
   ```

2. **Train on curated data from file-based datasets**

3. **Evaluate on torchvision datasets** instead:
   ```bash
   python experiments/03_model_customization.py \
       --input ./results/fer2013/curated/ \
       --config config.yaml \
       --target-dataset cifar10  # Use CIFAR-10 for evaluation instead
   ```

This allows you to use file-based data for training while evaluating on easier-to-access datasets.

---

## Verification Steps

After setting up file-based datasets, verify the structure:

```bash
# Check directory structure
ls -R data/elevater/{dataset_name}/

# Verify image counts per class
find data/elevater/{dataset_name}/test -type f -name "*.jpg" | wc -l

# Test loading in Python
python -c "
from maveric.datasets import get_dataset
dataset = get_dataset('fer2013', train=False, root='./data')
print(f'Test samples: {len(dataset._dataset) if dataset._dataset else 0}')
"
```

---

## Recommended Immediate Action

**Start with FER2013** (easiest to set up):

1. Download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Extract to `data/elevater/fer2013/` with train/test splits
3. Run test:
   ```bash
   python experiments/03_model_customization.py \
       --input ./results/fer2013/curated/ \
       --config experiments/maveric_config.yaml
   ```

If successful, follow the same pattern for other file-based datasets as needed.

---

## Summary

- **Issue**: File-based datasets don't auto-download like torchvision datasets
- **8 affected datasets**: FER2013, HatefulMemes, Kitti, PCAM, RenderedSST2, RESISC45, StanfordCars, VOC2007
- **Quick solution**: Manual download + directory organization (Option 1)
- **Best solution**: Implement PyTorch wrappers (Option 2) - future work
- **Workaround**: Use for retrieval only, evaluate on torchvision datasets (Option 3)

Choose Option 1 for immediate needs, or Option 3 if you only need file-based data for training purposes.
