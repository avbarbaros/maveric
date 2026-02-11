# Dataset-Specific Domain Adaptation for Unified Training

**Date**: February 11, 2026
**Feature**: Per-Dataset Domain Adaptation in Unified Training Mode
**Status**: ✅ **IMPLEMENTED**

---

## Overview

Implemented dataset-specific domain adaptation for unified training, allowing different domain adaptation settings for each dataset based on their native resolution and characteristics.

### Why This Matters

Different ELEVATER datasets have vastly different native resolutions:
- **CIFAR-10/100**: 32×32 pixels (very low resolution)
- **MNIST**: 28×28 pixels (very low resolution)
- **FER2013**: 48×48 pixels (low resolution)
- **Food101, DTD, Flowers102**: High resolution (256×256+)

Applying the same domain adaptation (e.g., downsampling to 32×32) to all datasets would be inappropriate. High-resolution datasets don't need aggressive downsampling, while low-resolution datasets benefit from it.

---

## Configuration Structure

### YAML Configuration (`maveric_config.yaml`)

```yaml
training:
  # Global domain adaptation defaults (used as fallback)
  use_domain_adaptation: false  # Default: disabled

  # Global parameters (blur, JPEG, downsample probabilities)
  domain_blur_probability: 0.5
  domain_blur_sigma_range: [0.5, 2.0]
  domain_jpeg_probability: 0.4
  domain_jpeg_quality_range: [50, 90]
  domain_downsample_probability: 0.7
  domain_target_size: null  # Default: use scale_range
  domain_downsample_scale_range: [0.5, 0.9]

  # Per-dataset overrides (only use_domain_adaptation and domain_target_size)
  dataset_domain_adaptation:
    # Low-resolution datasets - Enable with fixed target
    cifar10:
      use_domain_adaptation: true
      domain_target_size: 32
    cifar100:
      use_domain_adaptation: true
      domain_target_size: 32
    mnist:
      use_domain_adaptation: true
      domain_target_size: 28

    # Medium-resolution - Enable with scale range
    gtsrb:
      use_domain_adaptation: true
      domain_target_size: null  # Uses scale_range

    # High-resolution - Disable domain adaptation
    food101:
      use_domain_adaptation: false
    flowers102:
      use_domain_adaptation: false
```

---

## How It Works

### 1. Configuration Lookup Logic

For each sample during training:

```python
# Step 1: Get dataset name from sample
dataset_name = sample['source_dataset']  # e.g., 'cifar10'

# Step 2: Check if dataset has specific settings
if dataset_name in dataset_domain_adaptation:
    use_da = dataset_domain_adaptation[dataset_name]['use_domain_adaptation']
    target_size = dataset_domain_adaptation[dataset_name]['domain_target_size']
else:
    # Step 3: Fallback to global settings
    use_da = global_domain_config['use_domain_adaptation']
    target_size = global_domain_config['domain_target_size']

# Step 4: All other parameters (blur, JPEG, etc.) always from global config
```

### 2. Transform Application Order

```python
def __getitem__(self, idx):
    # 1. Load image from cache
    image = self.base_dataset._safe_get_image(url)

    # 2. Apply RandAugment (global augmentation)
    image = self.base_dataset._apply_transforms(image)

    # 3. Apply dataset-specific domain adaptation
    dataset_name = sample['source_dataset']
    image = self._apply_domain_adaptation(image, dataset_name)

    # 4. Return with global label
    return image, text, global_label
```

### 3. Domain Adaptation Transforms

**If enabled for dataset**:

1. **Gaussian Blur** (probability: `domain_blur_probability`)
   - Simulates low quality, pixelation, or motion blur
   - Sigma range: `domain_blur_sigma_range`

2. **JPEG Compression** (probability: `domain_jpeg_probability`)
   - Adds compression artifacts typical of web images
   - Quality range: `domain_jpeg_quality_range`

3. **Downsampling** (probability: `domain_downsample_probability`)
   - **Fixed target size** (if `domain_target_size` is specified):
     - Resize to target (e.g., 32×32), then resize back to original
     - Example: 224×224 → 32×32 → 224×224 (simulates CIFAR-10 resolution)
   - **Scale range** (if `domain_target_size` is `null`):
     - Resize by random scale factor, then resize back
     - Example: 224×224 → 157×157 → 224×224 (scale=0.7)

---

## Dataset Categorization

### Low-Resolution (Enable with Fixed Target)

**CIFAR-10/100** (32×32):
```yaml
cifar10:
  use_domain_adaptation: true
  domain_target_size: 32
```
- Native resolution: 32×32
- Benefit: Simulates test data resolution exactly

**MNIST** (28×28):
```yaml
mnist:
  use_domain_adaptation: true
  domain_target_size: 28
```
- Native resolution: 28×28
- Benefit: Matches MNIST's low resolution characteristics

### Medium-Resolution (Enable with Scale Range)

**GTSRB, EuroSAT, RESISC45** (variable):
```yaml
gtsrb:
  use_domain_adaptation: true
  domain_target_size: null  # Uses scale_range [0.5, 0.9]
```
- Native resolution: Variable (typically 64×64 to 128×128)
- Benefit: Simulates resolution degradation without fixed target

### Special Resolution Datasets

**FER2013** (48×48):
```yaml
fer2013:
  use_domain_adaptation: true
  domain_target_size: 48
```

**PatchCamelyon** (96×96):
```yaml
pcam:
  use_domain_adaptation: true
  domain_target_size: 96
```

### High-Resolution (Disable Domain Adaptation)

**Food101, DTD, Flowers102, Oxford Pets, etc.**:
```yaml
food101:
  use_domain_adaptation: false
```
- Native resolution: 256×256+ (high quality)
- Rationale: Test images are already high resolution, no need to simulate degradation

---

## Implementation Details

### Modified Files

**1. `maveric/customization/unified_training.py`**

**Added methods to `UnifiedELEVATERDataset`**:

```python
def _build_domain_transforms(self):
    """Build domain adaptation transforms for each dataset."""
    for dataset_name in self.dataset_metadata.keys():
        # Get dataset-specific or global settings
        dataset_config = self.dataset_domain_adaptation.get(dataset_name, {})
        use_da = dataset_config.get('use_domain_adaptation',
                                   self.global_domain_config.get('use_domain_adaptation', False))

        if use_da:
            self.dataset_transforms[dataset_name] = {
                'enabled': True,
                'target_size': dataset_config.get('domain_target_size', ...),
                'blur_prob': self.global_domain_config.get('domain_blur_probability', 0.5),
                # ... other global parameters
            }

def _apply_domain_adaptation(self, image, dataset_name: str):
    """Apply dataset-specific domain adaptation."""
    transform_config = self.dataset_transforms.get(dataset_name, {'enabled': False})

    if not transform_config['enabled']:
        return image

    # Apply blur, JPEG, downsampling based on config
    # ...
    return image
```

**Updated `__init__()` signature**:
```python
def __init__(self,
             unified_data: Dict,
             class_info: Dict,
             processor: CLIPProcessor,
             use_augmentation: bool = True,
             augmentation_config: Optional[Dict] = None,
             dataset_domain_adaptation: Optional[Dict] = None,  # NEW
             global_domain_config: Optional[Dict] = None,        # NEW
             cache_dir: Optional[str] = None):
```

**2. `experiments/03_model_customization.py`**

**Updated `run_unified_training()` function**:

```python
# Prepare dataset-specific domain adaptation settings
dataset_domain_adaptation = training_cfg.get('dataset_domain_adaptation', {})

# Prepare global domain adaptation config (used as fallback)
global_domain_config = {
    'use_domain_adaptation': training_cfg.get('use_domain_adaptation', False),
    'domain_blur_probability': training_cfg.get('domain_blur_probability', 0.5),
    'domain_blur_sigma_range': training_cfg.get('domain_blur_sigma_range', [0.5, 2.0]),
    'domain_jpeg_probability': training_cfg.get('domain_jpeg_probability', 0.4),
    'domain_jpeg_quality_range': training_cfg.get('domain_jpeg_quality_range', [50, 90]),
    'domain_downsample_probability': training_cfg.get('domain_downsample_probability', 0.7),
    'domain_target_size': training_cfg.get('domain_target_size', None),
    'domain_downsample_scale_range': training_cfg.get('domain_downsample_scale_range', [0.5, 0.9])
}

training_dataset = UnifiedELEVATERDataset(
    unified_data=unified_data,
    class_info=class_info,
    processor=processor,
    use_augmentation=training_cfg.get('use_augmentation', True),
    augmentation_config={...},
    dataset_domain_adaptation=dataset_domain_adaptation,  # NEW
    global_domain_config=global_domain_config,            # NEW
    cache_dir=config.get('cache_base_dir', './maveric_cache')
)
```

**3. `experiments/maveric_config.yaml`**

Added `dataset_domain_adaptation` section with 20 ELEVATER datasets categorized.

---

## Console Output

### During Training

```
📦 Creating unified training dataset...

📊 Unified dataset ready: 101,234 valid samples, 1,196 total classes

🎨 Domain Adaptation enabled for 7 datasets:
   • cifar10: target=32x32
   • cifar100: target=32x32
   • mnist: target=28x28
   • gtsrb: target=scale_range
   • eurosat: target=scale_range
   • fer2013: target=48x48
   • pcam: target=96x96
```

---

## Benefits

### 1. Appropriate Domain Matching

- **Low-resolution datasets** (CIFAR-10, MNIST): Simulates their native low resolution
- **High-resolution datasets** (Food101, Flowers102): Keeps high quality, no unnecessary degradation
- **Medium-resolution datasets** (GTSRB, EuroSAT): Flexible scaling without fixed target

### 2. Per-Sample Efficiency

- Domain adaptation applied **per-sample** during `__getitem__()`
- No need to create separate dataset loaders per dataset
- Efficient batching across all datasets

### 3. Easy Configuration

- Centralized configuration in YAML file
- Only need to specify `use_domain_adaptation` and `domain_target_size` per dataset
- All other parameters (blur, JPEG probabilities) are global

### 4. Fallback Mechanism

- Datasets not explicitly listed use global defaults
- No need to specify every dataset (only ones that need custom settings)
- Safe default: `use_domain_adaptation: false`

---

## Usage Examples

### Example 1: Default (No Domain Adaptation)

```yaml
training:
  use_domain_adaptation: false  # Global default
  dataset_domain_adaptation: {}  # Empty - no overrides
```

**Result**: No domain adaptation for any dataset.

### Example 2: Enable Only for Low-Resolution Datasets

```yaml
training:
  use_domain_adaptation: false  # Global default: disabled
  dataset_domain_adaptation:
    cifar10:
      use_domain_adaptation: true
      domain_target_size: 32
    cifar100:
      use_domain_adaptation: true
      domain_target_size: 32
    mnist:
      use_domain_adaptation: true
      domain_target_size: 28
```

**Result**: Domain adaptation only for CIFAR-10, CIFAR-100, MNIST. All other datasets: no domain adaptation.

### Example 3: Enable Globally, Disable for High-Resolution

```yaml
training:
  use_domain_adaptation: true  # Global default: enabled
  domain_target_size: null     # Use scale_range by default
  dataset_domain_adaptation:
    # Override low-resolution datasets with fixed targets
    cifar10:
      domain_target_size: 32
    mnist:
      domain_target_size: 28
    # Disable for high-resolution datasets
    food101:
      use_domain_adaptation: false
    flowers102:
      use_domain_adaptation: false
```

**Result**:
- CIFAR-10: Domain adaptation with target=32×32
- MNIST: Domain adaptation with target=28×28
- Food101, Flowers102: No domain adaptation
- All others: Domain adaptation with scale_range

---

## Testing Recommendations

### Visual Verification

```python
# Test dataset-specific domain adaptation
from maveric.customization.unified_training import UnifiedELEVATERDataset
import matplotlib.pyplot as plt

# Create dataset with domain adaptation
dataset = UnifiedELEVATERDataset(unified_data, class_info, processor, ...)

# Sample from different datasets
cifar10_idx = 0  # Assume first sample is CIFAR-10
food101_idx = 10000  # Assume later sample is Food101

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Sample CIFAR-10 images (should show downsampling to 32x32)
for i in range(5):
    image, text, label = dataset[cifar10_idx + i]
    axes[0, i].imshow(image)
    axes[0, i].set_title(f"CIFAR-10 (32x32)")

# Sample Food101 images (should show no downsampling)
for i in range(5):
    image, text, label = dataset[food101_idx + i]
    axes[1, i].imshow(image)
    axes[1, i].set_title(f"Food101 (high-res)")

plt.tight_layout()
plt.savefig("dataset_specific_domain_adaptation.png")
```

### Configuration Testing

```python
# Test 1: Verify dataset-specific settings are loaded
dataset = UnifiedELEVATERDataset(...)
assert dataset.dataset_transforms['cifar10']['enabled'] == True
assert dataset.dataset_transforms['cifar10']['target_size'] == 32
assert dataset.dataset_transforms['food101']['enabled'] == False

# Test 2: Verify fallback to global settings
# (For datasets not in dataset_domain_adaptation)
assert 'unknown_dataset' not in dataset.dataset_domain_adaptation
# Should use global setting
```

---

## Performance Considerations

### Runtime Overhead

**Per-sample domain adaptation adds**:
- Dictionary lookup: ~1 microsecond
- Conditional check: ~0.5 microseconds
- Transform application (if enabled): ~5-10 milliseconds (blur, JPEG, resize)

**Total overhead**: ~5-10ms per sample (only if domain adaptation enabled)

**For 100k samples**: ~8-17 minutes additional training time (acceptable)

### Memory Usage

**No additional memory overhead**:
- Domain transforms applied on-the-fly during `__getitem__()`
- No pre-computed transforms stored
- Transform config dict is negligible (~1KB)

---

## Comparison with Global Domain Adaptation

### Before (Global Domain Adaptation)

```python
# Single domain adaptation config for ALL datasets
training_dataset = UnifiedELEVATERDataset(
    ...,
    use_domain_adaptation=True,
    domain_adaptation_config={
        'domain_target_size': 32  # Applied to ALL datasets!
    }
)
```

**Problem**:
- Food101 images downsampled to 32×32 (inappropriate!)
- High-resolution datasets lose quality unnecessarily

### After (Dataset-Specific Domain Adaptation)

```python
# Per-dataset domain adaptation settings
training_dataset = UnifiedELEVATERDataset(
    ...,
    dataset_domain_adaptation={
        'cifar10': {'use_domain_adaptation': True, 'domain_target_size': 32},
        'food101': {'use_domain_adaptation': False}  # Keep high quality
    },
    global_domain_config={...}
)
```

**Solution**:
- CIFAR-10 images: Downsampled to 32×32 (appropriate!)
- Food101 images: Keep high resolution (appropriate!)

---

## Future Enhancements

### 1. Per-Dataset Blur/JPEG Probabilities

Currently, blur and JPEG probabilities are global. Could make them per-dataset:

```yaml
dataset_domain_adaptation:
  cifar10:
    use_domain_adaptation: true
    domain_target_size: 32
    domain_blur_probability: 0.7  # NEW - dataset-specific
    domain_jpeg_probability: 0.6  # NEW - dataset-specific
```

### 2. Automatic Target Size Detection

Auto-detect native resolution from dataset handler:

```python
def _auto_detect_target_size(self, dataset_name):
    """Auto-detect native resolution from dataset."""
    if dataset_name in ['cifar10', 'cifar100']:
        return 32
    elif dataset_name == 'mnist':
        return 28
    # ... etc
    return None  # Use scale_range
```

### 3. Domain Adaptation Scheduling

Apply different domain adaptation intensities during training:

```python
# Early epochs: Mild domain adaptation
# Later epochs: Stronger domain adaptation
epoch_factor = min(current_epoch / total_epochs, 1.0)
adjusted_prob = base_prob * epoch_factor
```

---

## Summary

✅ **Dataset-specific domain adaptation is fully implemented and ready for unified training!**

**Key Features**:
- ✅ Per-dataset `use_domain_adaptation` and `domain_target_size`
- ✅ Global fallback mechanism for unlisted datasets
- ✅ Centralized YAML configuration
- ✅ Per-sample transform application
- ✅ Console output showing enabled datasets
- ✅ Zero memory overhead

**Configuration**:
- 7 datasets with domain adaptation enabled (CIFAR-10/100, MNIST, GTSRB, EuroSAT, FER2013, PCAM)
- 13 datasets with domain adaptation disabled (high-resolution datasets)

**Usage**:
```bash
python experiments/03_model_customization.py \
    --unified-training \
    --input ./unified_training_data \
    --config experiments/maveric_config.yaml
```

🎉 **Ready for REACT-style unified training with appropriate per-dataset domain adaptation!**
