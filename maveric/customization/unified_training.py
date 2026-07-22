"""
Unified training utilities for REACT-style multi-dataset training.

This module provides functions for combining multiple ELEVATER datasets
into a single unified training session.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import CLIPProcessor

from ..datasets.elevater_datasets import ELEVATERDataset


def load_datasets_from_directory(input_dir: str) -> Dict[str, List[Path]]:
    """
    Load training data from manually prepared directory structure.

    Expected structure:
        input_dir/
        ├── cifar10/
        │   └── *training*maveric*.json
        ├── cifar100/
        │   └── *training*maveric*.json
        └── ...

    Args:
        input_dir: Directory containing dataset subdirectories

    Returns:
        Dict mapping dataset_name → list of JSON file paths

    Raises:
        ValueError: If input_dir doesn't exist or has no valid datasets

    Example:
        >>> files = load_datasets_from_directory('./unified_training_data')
        >>> print(files.keys())
        dict_keys(['cifar10', 'cifar100', 'caltech101', ...])
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
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        total_samples += len(data) if isinstance(data, list) else 1
                except Exception as e:
                    print(f"      ⚠️  Error reading {json_file.name}: {e}")
                    continue

            if total_samples > 0:
                dataset_files[dataset_name] = json_files
                print(f"   ✅ {dataset_name}: {len(json_files)} file(s), {total_samples:,} samples")
        else:
            print(f"   ⚠️  {dataset_name}: No training JSON files found, skipped")

    if not dataset_files:
        raise ValueError(
            f"No valid datasets found in {input_dir}. "
            f"Ensure each subdirectory contains *training*maveric*.json files."
        )

    print(f"\n📊 Total: {len(dataset_files)} datasets loaded")
    return dataset_files


def load_unified_dataset(
    dataset_files: Dict[str, List[Path]],
    max_samples_per_dataset: Optional[int] = None,
    seed: int = 42
) -> Dict:
    """
    Load and combine samples from multiple datasets.

    Each sample gets tagged with:
    - source_dataset: Original dataset name
    - dataset_idx: Numeric index for dataset (for tracking)

    Args:
        dataset_files: Dict mapping dataset_name → list of JSON file paths
        max_samples_per_dataset: Optional limit on samples per dataset
        seed: Random seed for reproducible sampling

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
                ...
            }
        }
    """
    all_samples = []
    dataset_metadata = {}
    current_idx = 0

    print("\n🔄 Loading and combining datasets...")

    for dataset_idx, (dataset_name, json_files) in enumerate(sorted(dataset_files.items())):
        # Load all JSON files for this dataset
        dataset_samples = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        dataset_samples.extend(data)
                    else:
                        dataset_samples.append(data)
            except Exception as e:
                print(f"      ⚠️  Error loading {json_file.name}: {e}")
                continue

        if not dataset_samples:
            print(f"   ⚠️  {dataset_name}: No samples loaded, skipped")
            continue

        # Apply max samples limit if specified
        if max_samples_per_dataset and len(dataset_samples) > max_samples_per_dataset:
            random.seed(seed)
            dataset_samples = random.sample(dataset_samples, max_samples_per_dataset)
            print(f"   ⚖️  {dataset_name}: Limited to {max_samples_per_dataset} samples")

        # Get class names from ELEVATER_DATASETS
        if dataset_name in ELEVATERDataset.ELEVATER_DATASETS:
            class_names = ELEVATERDataset.ELEVATER_DATASETS[dataset_name]['class_names']
            num_classes = ELEVATERDataset.ELEVATER_DATASETS[dataset_name]['num_classes']
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
            sample['dataset_idx'] = dataset_idx
            sample_indices.append(current_idx)
            all_samples.append(sample)
            current_idx += 1

        # Store metadata
        dataset_metadata[dataset_name] = {
            'num_samples': len(dataset_samples),
            'num_classes': num_classes,
            'class_names': class_names,
            'sample_indices': sample_indices,
            'dataset_idx': dataset_idx
        }

        print(f"   ✅ {dataset_name}: {len(dataset_samples):,} samples, {num_classes} classes")

    print(f"\n📊 Combined: {len(all_samples):,} total samples from {len(dataset_metadata)} datasets")

    return {
        'samples': all_samples,
        'dataset_metadata': dataset_metadata
    }


def unify_class_names(dataset_metadata: Dict) -> Dict:
    """
    Create unified class name mapping across datasets.

    Strategy:
    - Each dataset keeps its original class names
    - Create global class_id = (dataset_idx, local_class_idx)
    - Total classes = sum of all dataset classes
    - Prefix class names with dataset to avoid collisions

    Args:
        dataset_metadata: Metadata dict from load_unified_dataset()

    Returns:
        {
            'global_class_names': List[str],  # All prefixed class names
            'dataset_class_offsets': Dict[str, int],  # Starting index per dataset
            'num_total_classes': int  # Total number of classes
        }

    Example:
        >>> result = unify_class_names(metadata)
        >>> print(result['global_class_names'][:3])
        ['cifar10::airplane', 'cifar10::automobile', 'cifar10::bird']
        >>> print(result['dataset_class_offsets'])
        {'cifar10': 0, 'cifar100': 10, 'caltech101': 110, ...}
    """
    global_class_names = []
    dataset_class_offsets = {}
    current_offset = 0

    print("\n🔢 Creating unified class space...")

    for dataset_name, metadata in sorted(dataset_metadata.items()):
        dataset_class_offsets[dataset_name] = current_offset

        # Add prefixed class names
        for class_name in metadata['class_names']:
            # Handle list-based class names (FER2013)
            if isinstance(class_name, list):
                canonical_name = class_name[0]
            else:
                canonical_name = class_name

            prefixed_name = f"{dataset_name}::{canonical_name}"
            global_class_names.append(prefixed_name)

        print(f"   {dataset_name}: classes {current_offset}-{current_offset + metadata['num_classes'] - 1}")

        current_offset += metadata['num_classes']

    print(f"\n📊 Total unified classes: {len(global_class_names)}")

    return {
        'global_class_names': global_class_names,
        'dataset_class_offsets': dataset_class_offsets,
        'num_total_classes': len(global_class_names)
    }


def create_class_to_idx_mapping(
    dataset_metadata: Dict,
    dataset_class_offsets: Dict
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Create mappings from class names to indices.

    Args:
        dataset_metadata: Metadata for each dataset
        dataset_class_offsets: Starting offset for each dataset

    Returns:
        - global_class_to_idx: Maps "dataset::class" → global index
        - local_class_to_idx: Maps dataset → {class → local index}

    Example:
        >>> global_map, local_maps = create_class_to_idx_mapping(metadata, offsets)
        >>> print(global_map['cifar10::airplane'])
        0
        >>> print(local_maps['cifar10']['airplane'])
        0
    """
    global_class_to_idx = {}
    local_class_to_idx = {}

    for dataset_name, metadata in dataset_metadata.items():
        offset = dataset_class_offsets[dataset_name]
        local_map = {}

        for local_idx, class_name in enumerate(metadata['class_names']):
            # Handle list-based class names
            if isinstance(class_name, list):
                canonical_name = class_name[0]
            else:
                canonical_name = class_name

            global_idx = offset + local_idx
            global_key = f"{dataset_name}::{canonical_name}"

            global_class_to_idx[global_key] = global_idx
            local_map[canonical_name] = local_idx

        local_class_to_idx[dataset_name] = local_map

    return global_class_to_idx, local_class_to_idx


class UnifiedELEVATERDataset(torch.utils.data.Dataset):
    """
    Dataset that combines samples from multiple ELEVATER datasets.

    This dataset:
    - Loads samples from multiple datasets with source tagging
    - Maps local class labels to unified global class space
    - Handles image loading and caching
    - Applies augmentation and DATASET-SPECIFIC domain adaptation

    Args:
        unified_data: Dict from load_unified_dataset()
        class_info: Dict from unify_class_names()
        processor: CLIP processor for image preprocessing
        use_augmentation: Whether to apply data augmentation
        augmentation_config: RandAugment configuration
        dataset_domain_adaptation: Dict of dataset-specific domain adaptation settings
            Format: {dataset_name: {'use_domain_adaptation': bool, 'domain_target_size': int|None}}
        global_domain_config: Global domain adaptation config (used as fallback)
            Includes: use_domain_adaptation, domain_target_size, blur/jpeg/downsample settings
        cache_dir: Base directory for image caching

    Example:
        >>> unified_data = load_unified_dataset(dataset_files)
        >>> class_info = unify_class_names(unified_data['dataset_metadata'])
        >>> dataset_domain_adaptation = {
        ...     'cifar10': {'use_domain_adaptation': True, 'domain_target_size': 32},
        ...     'mnist': {'use_domain_adaptation': True, 'domain_target_size': 28}
        ... }
        >>> dataset = UnifiedELEVATERDataset(
        ...     unified_data, class_info, processor,
        ...     dataset_domain_adaptation=dataset_domain_adaptation
        ... )
        >>> print(len(dataset))
        102000
        >>> image, text, label = dataset[0]
        >>> print(label)  # Global label across all datasets
        42
    """

    def __init__(self,
                 unified_data: Dict,
                 class_info: Dict,
                 processor: CLIPProcessor,
                 use_augmentation: bool = True,
                 augmentation_config: Optional[Dict] = None,
                 dataset_domain_adaptation: Optional[Dict] = None,
                 global_domain_config: Optional[Dict] = None,
                 cache_dir: Optional[str] = None,
                 training_data_dir: Optional[str] = None):
        # Import here to avoid circular dependency
        from .model_customizer import LAIONCustomDataset

        self.dataset_metadata = unified_data['dataset_metadata']
        self.class_offsets = class_info['dataset_class_offsets']
        self.num_total_classes = class_info['num_total_classes']
        self.global_class_names = class_info['global_class_names']
        self.processor = processor

        # Store dataset-specific domain adaptation settings
        self.dataset_domain_adaptation = dataset_domain_adaptation or {}
        self.global_domain_config = global_domain_config or {}

        # Store unified samples
        self.unified_samples = unified_data['samples']

        # Store training data directory for dataset-specific images/ folders
        self.training_data_dir = Path(training_data_dir) if training_data_dir else None

        # Create a temporary LAIONCustomDataset WITHOUT domain adaptation
        # We'll apply dataset-specific domain adaptation in __getitem__
        self.base_dataset = LAIONCustomDataset(
            samples=unified_data['samples'],
            class_names=class_info['global_class_names'],
            processor=processor,
            use_augmentation=use_augmentation,
            augmentation_config=augmentation_config,
            use_domain_adaptation=False,  # Disabled - we'll apply per-dataset in __getitem__
            domain_adaptation_config=None,
            cache_dir=cache_dir,
            training_data_path=None,  # Use global cache for unified training
            skip_validation=True  # Skip slow validation - we'll do fast validation below
        )

        # Override validation to use dataset-specific images/ folders if available
        if self.training_data_dir:
            self._validate_from_dataset_folders(unified_data['samples'])

        # After filtering, we need to update our samples list
        self.valid_samples = self.base_dataset.valid_samples

        # Build dataset-specific domain transforms
        self._build_domain_transforms()

        print(f"\n📊 Unified dataset ready: {len(self.valid_samples):,} valid samples, {self.num_total_classes} total classes")
        self._print_domain_adaptation_summary()

    def _validate_from_dataset_folders(self, samples: List[Dict]):
        """
        Fast validation using dataset-specific images/ folders.
        Overrides LAIONCustomDataset's slow global cache validation.

        Strategy: Build in-memory index of all available images per dataset first,
        then validate against index. This avoids thousands of .exists() calls on Google Drive.
        """
        from PIL import Image
        from tqdm import tqdm
        import hashlib
        from collections import defaultdict

        print(f"✨ Using dataset-specific image folders for fast validation")
        print(f"   Base directory: {self.training_data_dir}")
        print(f"   Pattern: {{base_dir}}/{{dataset_name}}/images/{{hash}}.{{ext}}")

        # Step 1: Build in-memory index of available images per dataset
        print(f"\n📂 Building image index (one-time directory scan)...")
        image_index = defaultdict(set)  # dataset_name -> set of available filenames

        # Get unique datasets from samples
        unique_datasets = set(s.get('source_dataset') for s in samples if s.get('source_dataset'))
        print(f"   Found {len(unique_datasets)} unique datasets: {sorted(unique_datasets)}")

        for dataset_name in tqdm(unique_datasets, desc="Indexing datasets"):
            dataset_images_dir = self.training_data_dir / dataset_name / 'images'
            if dataset_images_dir.exists() and dataset_images_dir.is_dir():
                try:
                    # List all files once, store basenames without extension
                    for img_file in dataset_images_dir.iterdir():
                        if img_file.is_file():
                            # Store full filename (with extension) for lookup
                            image_index[dataset_name].add(img_file.name)
                    print(f"   ✓ {dataset_name}: {len(image_index[dataset_name])} images indexed")
                except Exception as e:
                    print(f"   ✗ {dataset_name}: Failed to index - {e}")
            else:
                print(f"   ✗ {dataset_name}: Directory not found at {dataset_images_dir}")

        # Step 2: Validate samples against index
        print(f"\n🔍 Validating {len(samples)} samples against image index...")

        valid_samples = []
        no_url_count = 0
        no_dataset_count = 0
        debug_shown = False

        for sample in tqdm(samples, desc="Validating samples"):
            url = sample.get('url')
            if not url:
                no_url_count += 1
                continue

            # Get dataset source from sample
            dataset_name = sample.get('source_dataset')
            if not dataset_name:
                no_dataset_count += 1
                continue

            # Check if we have this dataset indexed
            if dataset_name not in image_index:
                continue

            # Build expected filename hash
            url_hash = hashlib.md5(url.encode()).hexdigest()

            # Debug: Show what we're looking for vs what's available (first sample only)
            if not debug_shown and dataset_name in image_index and len(image_index[dataset_name]) > 0:
                print(f"\n🔍 Debug - First lookup:")
                print(f"   Dataset: {dataset_name}")
                print(f"   URL: {url[:80]}...")
                print(f"   Expected hash: {url_hash}")
                print(f"   Expected filenames: {url_hash}.jpg, {url_hash}.jpeg, etc.")
                print(f"   First 5 actual files: {list(image_index[dataset_name])[:5]}")
                debug_shown = True

            # Check if any file with this hash exists in our index
            # Note: Dataset-specific images/ folders use 'img_{hash}.jpg' format (copied from global cache)
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
                filename = f'img_{url_hash}{ext}'  # Add 'img_' prefix
                if filename in image_index[dataset_name]:
                    valid_samples.append(sample)
                    found = True
                    break

        # Override base_dataset's valid_samples
        self.base_dataset.valid_samples = valid_samples

        print(f"\n📊 Validation Summary:")
        print(f"   Total samples: {len(samples)}")
        print(f"   Samples without URL: {no_url_count}")
        print(f"   Samples without source_dataset: {no_dataset_count}")
        print(f"   Valid samples found: {len(valid_samples)}")
        print(f"   Success rate: {100*len(valid_samples)/len(samples):.1f}%")

    def _build_domain_transforms(self):
        """Build domain adaptation transforms for each dataset."""
        from PIL import Image, ImageFilter
        import io

        self.dataset_transforms = {}

        for dataset_name in self.dataset_metadata.keys():
            # Get dataset-specific settings or use global defaults
            dataset_config = self.dataset_domain_adaptation.get(dataset_name, {})
            use_da = dataset_config.get('use_domain_adaptation',
                                       self.global_domain_config.get('use_domain_adaptation', False))

            if use_da:
                target_size = dataset_config.get('domain_target_size',
                                                self.global_domain_config.get('domain_target_size', None))

                self.dataset_transforms[dataset_name] = {
                    'enabled': True,
                    'target_size': target_size,
                    'blur_prob': self.global_domain_config.get('domain_blur_probability', 0.5),
                    'blur_sigma_range': self.global_domain_config.get('domain_blur_sigma_range', [0.5, 2.0]),
                    'jpeg_prob': self.global_domain_config.get('domain_jpeg_probability', 0.4),
                    'jpeg_quality_range': self.global_domain_config.get('domain_jpeg_quality_range', [50, 90]),
                    'downsample_prob': self.global_domain_config.get('domain_downsample_probability', 0.7),
                    'scale_range': self.global_domain_config.get('domain_downsample_scale_range', [0.5, 0.9])
                }
            else:
                self.dataset_transforms[dataset_name] = {'enabled': False}

    def _print_domain_adaptation_summary(self):
        """Print summary of domain adaptation settings per dataset."""
        enabled_datasets = [name for name, config in self.dataset_transforms.items() if config['enabled']]
        if enabled_datasets:
            print(f"\n🎨 Domain Adaptation enabled for {len(enabled_datasets)} datasets:")
            for dataset_name in enabled_datasets:
                config = self.dataset_transforms[dataset_name]
                target = f"{config['target_size']}x{config['target_size']}" if config['target_size'] else "scale_range"
                print(f"   • {dataset_name}: target={target}")

    def _apply_domain_adaptation(self, image, dataset_name: str):
        """Apply dataset-specific domain adaptation to image."""
        import random
        from PIL import Image, ImageFilter
        import io

        transform_config = self.dataset_transforms.get(dataset_name, {'enabled': False})

        if not transform_config['enabled']:
            return image

        # Apply Gaussian blur
        if random.random() < transform_config['blur_prob']:
            sigma = random.uniform(*transform_config['blur_sigma_range'])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Apply JPEG compression
        if random.random() < transform_config['jpeg_prob']:
            quality = random.randint(*transform_config['jpeg_quality_range'])
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).copy()

        # Apply downsampling
        if random.random() < transform_config['downsample_prob']:
            orig_w, orig_h = image.size

            if transform_config['target_size'] is not None:
                # Fixed target size
                new_size = transform_config['target_size']
                image = image.resize((new_size, new_size), Image.BILINEAR)
                image = image.resize((orig_w, orig_h), Image.BILINEAR)
            else:
                # Scale range
                scale = random.uniform(*transform_config['scale_range'])
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                image = image.resize((new_w, new_h), Image.BILINEAR)
                image = image.resize((orig_w, orig_h), Image.BILINEAR)

        return image

    def __len__(self):
        return len(self.valid_samples)

    def _load_dataset_image(self, url, dataset_name):
        """
        Load an image from its dataset-specific images/ folder
        ({training_data_dir}/{dataset_name}/images/img_{hash}.{ext}), matching
        the lookup _validate_from_dataset_folders() used to confirm the sample
        is valid. Falls back to the base dataset's global cache if the
        dataset-specific folder isn't available for some reason.
        """
        from PIL import Image

        if url and self.training_data_dir:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            images_dir = self.training_data_dir / dataset_name / 'images'
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
                candidate = images_dir / f'img_{url_hash}{ext}'
                if candidate.exists():
                    try:
                        image = Image.open(candidate)
                        image.load()
                        return image.convert('RGB')
                    except Exception:
                        break  # Fall through to global cache

        return self.base_dataset._safe_get_image(url)

    def __getitem__(self, idx):
        """
        Get a training sample with global label and dataset-specific domain adaptation.

        Returns:
            tuple: (image, text, global_label)
                - image: PIL Image (transformed with dataset-specific domain adaptation)
                - text: Caption string
                - global_label: Label in global class space (0 to num_total_classes-1)
        """
        # Get the sample
        sample = self.valid_samples[idx]

        # Get dataset name for dataset-specific transforms
        dataset_name = sample['source_dataset']

        # Load from the dataset-specific images/ folder (same location
        # _validate_from_dataset_folders already confirmed the file exists in),
        # not the global cache. base_dataset._safe_get_image() only knows about
        # the global cache (training_data_path=None was passed to it above), so
        # using it here silently misses the fast local copy on every access.
        image = self._load_dataset_image(sample.get('url'), dataset_name)
        image = self.base_dataset._apply_transforms(image)  # Applies augmentation

        # Apply dataset-specific domain adaptation AFTER augmentation
        image = self._apply_domain_adaptation(image, dataset_name)

        # Get local label from sample
        sample_label = sample['label']

        # Convert local label to global label
        dataset_metadata = self.dataset_metadata[dataset_name]
        class_names = dataset_metadata['class_names']

        # Find local index
        # Normalize label for matching
        normalized_label = self.base_dataset._normalize_label(sample_label)
        local_idx = None
        for i, class_name in enumerate(class_names):
            if isinstance(class_name, list):
                canonical_name = class_name[0]
            else:
                canonical_name = class_name
            if self.base_dataset._normalize_label(canonical_name) == normalized_label:
                local_idx = i
                break

        if local_idx is None:
            # Fallback: use first class
            local_idx = 0

        # Map to global label
        offset = self.class_offsets[dataset_name]
        global_label = offset + local_idx

        return image, sample.get('text', ''), global_label


def evaluate_unified_model_per_dataset(
    model,
    dataset_metadata: Dict,
    class_offsets: Dict,
    processor: CLIPProcessor,
    device: str = 'cuda',
    batch_size: int = 32,
    use_templates: bool = True,
    cache_base_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate unified model separately on each dataset's test set.

    Delegates to ModelCustomizer._create_test_loader() which already handles:
    - torchvision vs file-based datasets
    - class name normalisation (REACT / ELEVATER exact names)
    - ImageFolder fallback for file-based datasets
    - FER2013 list-based class names

    Args:
        model: Trained unified model (CustomizedCLIP)
        dataset_metadata: Metadata dict from load_unified_dataset()
        class_offsets: Class offset mapping from unify_class_names()
        processor: CLIP processor
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        use_templates: Whether to use REACT-style templates
        cache_base_dir: Base directory for dataset caching

    Returns:
        Dict mapping dataset_name → accuracy
    """
    from ..datasets.elevater_datasets import ELEVATERDataset
    from ..customization.evaluation import Evaluator
    from ..customization.model_customizer import ModelCustomizer

    results = {}
    model.eval()

    print("\n📊 Evaluating unified model on each dataset separately...")

    # Build a lightweight ModelCustomizer that carries our trained model/processor
    # so we can reuse _create_test_loader without re-downloading any model.
    customizer = ModelCustomizer.__new__(ModelCustomizer)  # bypass __init__
    from ..core.base import BaseComponent
    BaseComponent.__init__(customizer, "ModelCustomizer")
    customizer.base_model_name = "unified"
    customizer.device = device
    customizer.checkpoint_dir = None
    customizer.cache_base_dir = cache_base_dir
    customizer.model = model       # our trained weights
    customizer.processor = processor
    customizer.trainer = None
    customizer.evaluator = None

    for dataset_name, metadata in sorted(dataset_metadata.items()):
        print(f"\n🔍 Evaluating on {dataset_name}...")

        try:
            # Authoritative REACT class names from ELEVATER_DATASETS
            if dataset_name in ELEVATERDataset.ELEVATER_DATASETS:
                class_names = ELEVATERDataset.ELEVATER_DATASETS[dataset_name]['class_names']
            else:
                class_names = metadata['class_names']

            # Reuse the battle-tested test loader from ModelCustomizer
            test_loader = customizer._create_test_loader(
                target_dataset_name=dataset_name,
                class_names=class_names
            )

            if test_loader is None:
                print(f"   ⚠️  {dataset_name}: Test data not available, skipped")
                continue

            # Get REACT-style templates
            templates = None
            if use_templates:
                try:
                    dataset_handler = ELEVATERDataset(dataset_name, train=False)
                    templates = dataset_handler.get_text_templates()
                except Exception:
                    pass

            evaluator = Evaluator(
                model=model,
                processor=processor,
                class_names=class_names,
                device=device,
                templates=templates
            )

            accuracy = evaluator.evaluate(test_loader)
            num_samples = len(test_loader.dataset)

            results[dataset_name] = accuracy
            print(f"   ✅ {dataset_name}: {accuracy:.2%} accuracy ({num_samples:,} test samples)")

        except Exception as e:
            print(f"   ❌ {dataset_name}: Evaluation failed - {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n📊 Evaluation complete: {len(results)}/{len(dataset_metadata)} datasets evaluated")
    return results


def save_unified_results(
    results: Dict[str, float],
    output_dir: str,
    filename: str = "unified_training_results.json"
) -> str:
    """
    Save per-dataset evaluation results to JSON file.

    Args:
        results: Dict mapping dataset_name → accuracy
        output_dir: Output directory
        filename: Output filename (default: unified_training_results.json)

    Returns:
        Path to saved file

    Example:
        >>> save_unified_results(
        ...     results={'cifar10': 0.92, 'cifar100': 0.68},
        ...     output_dir='./results/unified_training'
        ... )
        './results/unified_training/unified_training_results.json'
    """
    from pathlib import Path
    import json
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / filename

    # Create output structure
    output_data = {
        'evaluation_date': datetime.now().isoformat(),
        'num_datasets': len(results),
        'average_accuracy': sum(results.values()) / len(results) if results else 0.0,
        'per_dataset_results': results,
        'sorted_by_accuracy': sorted(
            results.items(),
            key=lambda x: x[1],
            reverse=True
        )
    }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"   Average accuracy: {output_data['average_accuracy']:.2%}")
    print(f"   Best dataset: {output_data['sorted_by_accuracy'][0][0]} ({output_data['sorted_by_accuracy'][0][1]:.2%})")
    print(f"   Worst dataset: {output_data['sorted_by_accuracy'][-1][0]} ({output_data['sorted_by_accuracy'][-1][1]:.2%})")

    return str(output_file)
