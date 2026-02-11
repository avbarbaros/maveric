# MAVERIC Unified Training Design - REACT-Style Multi-Dataset Training

## Executive Summary

**Goal**: Enable training a single CLIP model on combined data from ALL ELEVATER datasets (like REACT's 10M sample approach), while preserving individual dataset evaluation capabilities.

**Key Advantage**: One training session instead of 20, potentially better cross-dataset generalization.

**Your Achievement**: You've already surpassed REACT with individual dataset customization. This feature would enable comparison of both approaches.

---

## Current vs. Proposed Architecture

### Current Approach (Per-Dataset Training)
```
Dataset 1 → Retrieve → Curate → Train Model 1 → Evaluate on Dataset 1
Dataset 2 → Retrieve → Curate → Train Model 2 → Evaluate on Dataset 2
...
Dataset 20 → Retrieve → Curate → Train Model 20 → Evaluate on Dataset 20

Result: 20 specialized models, each optimized for one dataset
```

### Proposed Approach (REACT-Style Unified Training)
```
Dataset 1 → Retrieve → Curate ↘
Dataset 2 → Retrieve → Curate → User Organizes → Combine All → Train Single Model → Evaluate on Each
...                             ↗
Dataset 20 → Retrieve → Curate ↗

Result: 1 generalist model evaluated on all 20 datasets
```

---

## Design Specifications

### 1. Command-Line Interface

#### New Flag: `--unified-training`

```bash
# Example 1: Unified training with manually prepared directory
python experiments/03_model_customization.py \
    --unified-training \
    --input ./unified_training_data \
    --config experiments/maveric_config.yaml \
    --output-dir ./results/unified_model/

# Example 2: Traditional per-dataset training (existing behavior)
python experiments/03_model_customization.py \
    --input ./results/cifar10/curated/ \
    --config experiments/maveric_config.yaml
```

#### New Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--unified-training` | flag | Enable REACT-style unified training | False |
| `--input` | str | **Directory** containing dataset subdirectories | Required |
| `--max-samples-per-dataset` | int | Limit samples per dataset (for balance) | None |
| `--save-individual-results` | flag | Save per-dataset results separately | True |

**Key Change**: When `--unified-training` is set, `--input` expects a **directory containing dataset subdirectories**, not a single dataset's curated folder.

---

### 2. Input Directory Structure (Manual Preparation)

**User must manually prepare a unified training directory** with this exact structure:

```
unified_training_data/
├── cifar10/
│   ├── cifar10_training_maveric_1.json
│   └── cifar10_training_maveric_2.json
├── cifar100/
│   ├── cifar100_training_maveric_1.json
│   └── cifar100_training_maveric_2.json
├── caltech101/
│   └── caltech101_training_maveric_1.json
├── dtd/
│   └── dtd_training_maveric_1.json
├── eurosat/
│   └── eurosat_training_maveric_1.json
├── fer2013/
│   └── fer2013_training_maveric_1.json
├── fgvc_aircraft/
│   └── fgvc_aircraft_training_maveric_1.json
├── food101/
│   └── food101_training_maveric_1.json
├── gtsrb/
│   └── gtsrb_training_maveric_1.json
├── hateful_memes/
│   └── hateful_memes_training_maveric_1.json
├── kitti_distance/
│   └── kitti_distance_training_maveric_1.json
├── mnist/
│   └── mnist_training_maveric_1.json
├── oxford_flowers102/
│   └── oxford_flowers102_training_maveric_1.json
├── oxford_pets/
│   └── oxford_pets_training_maveric_1.json
├── patch_camelyon/
│   └── patch_camelyon_training_maveric_1.json
├── rendered_sst2/
│   └── rendered_sst2_training_maveric_1.json
├── resisc45/
│   └── resisc45_training_maveric_1.json
├── stanford_cars/
│   └── stanford_cars_training_maveric_1.json
├── country211/
│   └── country211_training_maveric_1.json
└── voc2007/
    └── voc2007_training_maveric_1.json
```

**Key Requirements**:
- Each subdirectory must be named exactly as the dataset name (lowercase with underscores)
- Each subdirectory must contain at least one JSON file matching `*training*maveric*.json` pattern
- User is responsible for copying/organizing the training data files
- Missing datasets will be skipped with a warning

**Example Setup Script**:
```bash
#!/bin/bash
# setup_unified_training.sh

# Create unified training directory
mkdir -p unified_training_data

# List of all ELEVATER datasets
DATASETS=(
    "cifar10" "cifar100" "caltech101" "dtd" "eurosat"
    "fer2013" "fgvc_aircraft" "food101" "gtsrb" "hateful_memes"
    "kitti_distance" "mnist" "oxford_flowers102" "oxford_pets"
    "patch_camelyon" "rendered_sst2" "resisc45" "stanford_cars"
    "country211" "voc2007"
)

# Copy curated data for each dataset
for dataset in "${DATASETS[@]}"; do
    if [ -d "results/${dataset}/curated" ]; then
        mkdir -p "unified_training_data/${dataset}"
        cp results/${dataset}/curated/*training*maveric*.json "unified_training_data/${dataset}/"
        echo "✅ Copied ${dataset}"
    else
        echo "⚠️  Skipped ${dataset} (no curated data found)"
    fi
done

echo "✅ Unified training directory prepared!"
echo "📊 Run: python experiments/03_model_customization.py --unified-training --input ./unified_training_data"
```

---

### 3. Data Loading & Combination Strategy

#### Step 1: Load from Expected Structure
```python
def load_datasets_from_directory(input_dir: str) -> Dict[str, List[Path]]:
    """
    Load training data from manually prepared directory structure.

    Args:
        input_dir: Directory containing dataset subdirectories

    Returns:
        Dict mapping dataset_name → list of JSON file paths

    Example:
        {
            'cifar10': [Path('unified_training_data/cifar10/cifar10_training_maveric_1.json')],
            'cifar100': [Path('unified_training_data/cifar100/cifar100_training_maveric_1.json')],
            ...
        }
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not input_path.is_dir():
        raise ValueError(f"Input path must be a directory when using --unified-training: {input_dir}")

    dataset_files = {}

    print(f"🔍 Loading datasets from: {input_dir}")

    # Scan for dataset subdirectories
    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue

        dataset_name = subdir.name

        # Find JSON files in subdirectory
        json_files = list(subdir.glob("*training*maveric*.json"))

        if json_files:
            # Count total samples
            total_samples = 0
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    total_samples += len(data) if isinstance(data, list) else 1

            dataset_files[dataset_name] = json_files
            print(f"   ✅ {dataset_name}: {len(json_files)} file(s), {total_samples:,} samples")
        else:
            print(f"   ⚠️  {dataset_name}: No training JSON files found, skipped")

    if not dataset_files:
        raise ValueError(f"No valid datasets found in {input_dir}. "
                        f"Ensure each subdirectory contains *training*maveric*.json files.")

    print(f"\n📊 Total: {len(dataset_files)} datasets loaded")
    return dataset_files
```

#### Step 2: Load & Tag Samples
```python
def load_unified_dataset(dataset_files: Dict[str, List[Path]],
                        max_samples_per_dataset: int = None) -> Dict:
    """
    Load and combine samples from multiple datasets.

    Each sample gets tagged with:
    - source_dataset: Original dataset name
    - dataset_idx: Numeric index for dataset (for tracking)

    Returns:
        {
            'samples': List[Dict],  # Combined samples with source_dataset tag
            'dataset_metadata': {
                'cifar10': {
                    'num_samples': 5000,
                    'num_classes': 10,
                    'class_names': ['airplane', 'automobile', ...],
                    'sample_indices': [0, 1, 2, ...]  # Indices in combined list
                },
                'cifar100': {
                    'num_samples': 5000,
                    'num_classes': 100,
                    'class_names': [...],
                    'sample_indices': [5000, 5001, ...]
                },
                ...
            }
        }
    """
    from maveric.datasets.elevater_datasets import ELEVATER_DATASETS

    all_samples = []
    dataset_metadata = {}
    current_idx = 0

    print("\n🔄 Loading and combining datasets...")

    for dataset_name, json_files in dataset_files.items():
        # Load all JSON files for this dataset
        dataset_samples = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset_samples.extend(data)
                else:
                    dataset_samples.append(data)

        # Apply max samples limit if specified
        if max_samples_per_dataset and len(dataset_samples) > max_samples_per_dataset:
            import random
            random.seed(42)  # Reproducible sampling
            dataset_samples = random.sample(dataset_samples, max_samples_per_dataset)
            print(f"   ⚖️  {dataset_name}: Limited to {max_samples_per_dataset} samples")

        # Get class names from ELEVATER_DATASETS
        if dataset_name in ELEVATER_DATASETS:
            class_names = ELEVATER_DATASETS[dataset_name]['class_names']
            num_classes = ELEVATER_DATASETS[dataset_name]['num_classes']
        else:
            # Fallback: extract from data
            print(f"   ⚠️  {dataset_name}: Not in ELEVATER_DATASETS, extracting classes from data")
            unique_labels = set(sample.get('label', '') for sample in dataset_samples)
            class_names = sorted(list(unique_labels))
            num_classes = len(class_names)

        # Tag samples with source dataset
        sample_indices = []
        for sample in dataset_samples:
            sample['source_dataset'] = dataset_name
            sample['dataset_idx'] = len(dataset_metadata)  # Numeric index
            sample_indices.append(current_idx)
            all_samples.append(sample)
            current_idx += 1

        # Store metadata
        dataset_metadata[dataset_name] = {
            'num_samples': len(dataset_samples),
            'num_classes': num_classes,
            'class_names': class_names,
            'sample_indices': sample_indices,
            'dataset_idx': len(dataset_metadata)
        }

        print(f"   ✅ {dataset_name}: {len(dataset_samples):,} samples, {num_classes} classes")

    print(f"\n📊 Combined: {len(all_samples):,} total samples from {len(dataset_metadata)} datasets")

    return {
        'samples': all_samples,
        'dataset_metadata': dataset_metadata
    }
```

#### Step 3: Class Name Unification
```python
def unify_class_names(dataset_metadata: Dict) -> Dict:
    """
    Create unified class name mapping across datasets.

    Strategy:
    - Each dataset keeps its original class names
    - Create global class_id = (dataset_idx, local_class_idx)
    - Total classes = sum of all dataset classes
    - Prefix class names with dataset to avoid collisions

    Example:
        CIFAR-10 has 10 classes (indices 0-9)
        CIFAR-100 has 100 classes (indices 10-109)
        Caltech101 has 102 classes (indices 110-211)
        ...

    Returns:
        {
            'global_class_names': [
                'cifar10::airplane',
                'cifar10::automobile',
                ...
                'cifar100::apple',
                ...
            ],
            'dataset_class_offsets': {
                'cifar10': 0,
                'cifar100': 10,
                'caltech101': 110,
                ...
            },
            'num_total_classes': 1196
        }
    """
    global_class_names = []
    dataset_class_offsets = {}
    current_offset = 0

    print("\n🔢 Creating unified class space...")

    for dataset_name, metadata in dataset_metadata.items():
        dataset_class_offsets[dataset_name] = current_offset

        # Add prefixed class names
        for class_name in metadata['class_names']:
            # Handle list-based class names (FER2013)
            if isinstance(class_name, list):
                class_name = class_name[0]

            prefixed_name = f"{dataset_name}::{class_name}"
            global_class_names.append(prefixed_name)

        print(f"   {dataset_name}: classes {current_offset}-{current_offset + metadata['num_classes'] - 1}")

        current_offset += metadata['num_classes']

    print(f"\n📊 Total unified classes: {len(global_class_names)}")

    return {
        'global_class_names': global_class_names,
        'dataset_class_offsets': dataset_class_offsets,
        'num_total_classes': len(global_class_names)
    }
```

---

### 4. Training Modifications

#### Unified Dataset Class
```python
class UnifiedELEVATERDataset(LAIONCustomDataset):
    """
    Dataset that combines samples from multiple ELEVATER datasets.

    Key features:
    - Maps local class labels to global class space
    - Tracks source dataset for each sample
    - Handles different class distributions
    """

    def __init__(self,
                 unified_data: Dict,
                 class_info: Dict,
                 processor: CLIPProcessor,
                 **kwargs):
        self.dataset_metadata = unified_data['dataset_metadata']
        self.class_offsets = class_info['dataset_class_offsets']
        self.num_total_classes = class_info['num_total_classes']
        self.global_class_names = class_info['global_class_names']

        # Parent init with unified samples
        super().__init__(
            samples=unified_data['samples'],
            class_names=class_info['global_class_names'],
            processor=processor,
            **kwargs
        )

    def __getitem__(self, idx):
        # Get sample (returns image, text, local_label)
        image, text, local_label = super().__getitem__(idx)

        # Convert local label to global label
        sample = self.valid_samples[idx]
        dataset_name = sample['source_dataset']
        offset = self.class_offsets[dataset_name]
        global_label = offset + local_label

        return image, text, global_label
```

---

### 5. Evaluation Strategy

#### Per-Dataset Evaluation
```python
def evaluate_unified_model_per_dataset(
    model,
    dataset_metadata: Dict,
    class_offsets: Dict,
    test_data_root: str
) -> Dict[str, Dict]:
    """
    Evaluate unified model on each dataset separately.

    For each dataset:
    1. Load test data for that dataset only
    2. Create text classifier using ONLY that dataset's classes
    3. Compute predictions in dataset's class space
    4. Calculate accuracy for that dataset

    Args:
        model: Trained unified model
        dataset_metadata: Metadata for each dataset
        class_offsets: Class offset mapping
        test_data_root: Root directory for test data

    Returns:
        {
            'cifar10': {
                'accuracy': 92.3,
                'class_accuracies': {...},
                'num_test_samples': 10000
            },
            'cifar100': {...},
            ...
        }
    """
    results = {}

    print("\n📊 Evaluating unified model on each dataset...")

    for dataset_name, metadata in dataset_metadata.items():
        print(f"\n   Evaluating {dataset_name}...")

        # Get dataset-specific info
        class_names = metadata['class_names']

        # Get templates for this dataset
        from maveric.datasets.elevater_datasets import ELEVATER_DATASETS
        if dataset_name in ELEVATER_DATASETS:
            templates = ELEVATER_DATASETS[dataset_name].get('templates', None)
        else:
            templates = None

        # Create test loader for this dataset
        test_loader = create_test_loader_for_dataset(
            dataset_name=dataset_name,
            class_names=class_names,
            test_data_root=test_data_root
        )

        if test_loader is None:
            print(f"      ⚠️  No test data available, skipped")
            continue

        # Evaluate on this dataset
        accuracy, class_accs = model.evaluator.evaluate_detailed(
            model=model,
            data_loader=test_loader,
            class_names=class_names,
            templates=templates
        )

        results[dataset_name] = {
            'accuracy': accuracy,
            'class_accuracies': class_accs,
            'num_test_samples': len(test_loader.dataset)
        }

        print(f"      ✅ Accuracy: {accuracy:.2f}%")

    return results
```

---

### 6. Usage Examples

#### Example 1: Prepare Unified Training Directory
```bash
# Manual approach: Create directory and copy files
mkdir -p unified_training_data

# Copy each dataset's curated data
cp -r results/cifar10/curated unified_training_data/cifar10
cp -r results/cifar100/curated unified_training_data/cifar100
cp -r results/caltech101/curated unified_training_data/caltech101
# ... continue for all datasets

# Verify structure
tree unified_training_data -L 2
```

#### Example 2: Run Unified Training
```bash
python experiments/03_model_customization.py \
    --unified-training \
    --input ./unified_training_data \
    --config experiments/maveric_config.yaml \
    --output-dir ./results/unified_model/ \
    --save-individual-results
```

#### Example 3: Run with Sample Balancing
```bash
# Limit each dataset to 1000 samples
python experiments/03_model_customization.py \
    --unified-training \
    --input ./unified_training_data \
    --max-samples-per-dataset 1000 \
    --config experiments/maveric_config.yaml \
    --output-dir ./results/unified_balanced/
```

---

### 7. Expected Console Output

```
🚀 Starting MAVERIC Model Customization (Unified Training Mode)...
📁 Input directory: ./unified_training_data
📋 Configuration file: maveric_config.yaml
✅ Configuration loaded from: experiments/maveric_config.yaml

🔍 Loading datasets from: ./unified_training_data
   ✅ caltech101: 1 file(s), 1,027 samples
   ✅ cifar10: 2 file(s), 5,000 samples
   ✅ cifar100: 2 file(s), 5,000 samples
   ✅ dtd: 1 file(s), 470 samples
   ✅ eurosat: 1 file(s), 500 samples
   ... (15 more datasets)

📊 Total: 20 datasets loaded

🔄 Loading and combining datasets...
   ✅ caltech101: 1,027 samples, 102 classes
   ✅ cifar10: 5,000 samples, 10 classes
   ✅ cifar100: 5,000 samples, 100 classes
   ... (17 more)

📊 Combined: 102,000 total samples from 20 datasets

🔢 Creating unified class space...
   caltech101: classes 0-101
   cifar10: classes 102-111
   cifar100: classes 112-211
   ... (17 more)

📊 Total unified classes: 1,196

🤖 Training unified model...
   Epochs: 20
   Learning rate: 7e-07
   Total classes: 1,196

📊 Evaluating unified model on each dataset...
   Evaluating caltech101...
      ✅ Accuracy: 85.2%
   Evaluating cifar10...
      ✅ Accuracy: 92.3%
   ... (18 more)

✅ Unified model training complete!
📊 Mean accuracy: 78.5%
📊 Mean improvement: +2.8%
```

---

## Summary

This simplified design gives users **explicit control** over the unified training process:

✅ **Manual directory preparation** - No auto-discovery confusion
✅ **Clear structure** - Simple subdirectory per dataset
✅ **User responsibility** - Explicitly copy/organize files
✅ **Validation** - Clear errors if structure is wrong
✅ **Flexible** - Include only datasets you want

**Next step**: Implement the `load_datasets_from_directory()` and modify `03_model_customization.py` to handle `--unified-training` flag!
