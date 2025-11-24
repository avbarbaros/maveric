# Reference Generation in MAVERIC

This document provides a comprehensive explanation of how MAVERIC generates and uses reference data for dataset retrieval and model customization.

## Overview

Reference generation is a **one-time setup process per target dataset** that creates baseline embeddings for similarity matching during data retrieval. MAVERIC generates two types of references:

1. **Reference Images**: Sample images from each class of the target dataset
2. **Reference Texts**: Dataset-specific text templates following REACT-style prompting

These references are cached and reused across multiple retrieval runs, providing **80-95% speedup** compared to regenerating them each time.

---

## Reference Generation Pipeline

### Step 1: Sample Selection

**Purpose**: Select representative images from the target dataset for each class.

**Process**:
1. Load the target dataset (e.g., CIFAR-10, Oxford Pets, EuroSAT)
2. For each class in the dataset:
   - Randomly sample `n_reference_images` images (default: 10 per class)
   - Ensure diverse sampling across the class
3. Save reference images to cache directory

**Configuration**:
```yaml
n_reference_images: 10  # Number of reference images per class
```

**Example Output** (for CIFAR-10 with 10 classes):
```
reference_images/
├── cifar10/
│   ├── airplane/
│   │   ├── ref_000.jpg
│   │   ├── ref_001.jpg
│   │   └── ... (10 images total)
│   ├── automobile/
│   │   ├── ref_000.jpg
│   │   └── ...
│   └── ... (10 classes total)
```

**Implementation**: [elevater_datasets.py](../maveric/datasets/elevater_datasets.py)

---

### Step 2: Text Template Generation

**Purpose**: Create dataset-specific text prompts that describe each class using REACT-style formatting.

**REACT-Style Prompting**:
MAVERIC implements context-aware text templates that match the visual domain of each dataset. This follows the REACT benchmark's approach of using domain-appropriate descriptions.

#### Template Categories

**Generic Datasets** (CIFAR-10, CIFAR-100, Caltech101):
```python
templates = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}."
]
```

**Texture Datasets** (DTD):
```python
templates = [
    "a photo of a {} texture.",
    "a close-up photo of a {} texture.",
    "{} texture."
]
```

**Satellite Imagery** (EuroSAT):
```python
templates = [
    "a centered satellite photo of {}.",
    "a satellite photo of {}."
]
```

**Traffic Signs** (GTSRB):
```python
templates = [
    "a zoomed in photo of a {} traffic sign.",
    "a centered photo of a {} traffic sign.",
    "a photo of a {} traffic sign."
]
```

**Food Items** (Food101):
```python
templates = [
    "a photo of {}, a type of food.",
    "a photo of {} food."
]
```

**Flowers** (Oxford Flowers102):
```python
templates = [
    "a photo of a {}, a type of flower.",
    "a close-up photo of a {} flower."
]
```

**Pets** (Oxford Pets):
```python
templates = [
    "a photo of a {}, a type of pet.",
    "a photo of a {}.",
    "a {} looking at the camera."
]
```

**Facial Expressions** (FER2013):
```python
templates = [
    "a photo of a {} face.",
    "a face with a {} expression.",
    "a {} facial expression."
]
```

#### Class Name Formatting

MAVERIC intelligently formats class names to fit natural language templates:

**Original Class Names** (from dataset):
- `speed_limit_30` (GTSRB)
- `beware_of_ice_snow` (GTSRB)
- `american_bulldog` (Oxford Pets)
- `Abyssinian` (Oxford Pets - from REACT)

**Formatted for Templates**:
- `speed limit 30` → "a photo of a speed limit 30 traffic sign"
- `beware of ice snow` → "a photo of a beware of ice snow traffic sign"
- `american bulldog` → "a photo of a american bulldog, a type of pet"
- `Abyssinian` → "a photo of a Abyssinian, a type of pet"

**Normalization Rules**:
1. Replace underscores with spaces: `speed_limit_30` → `speed limit 30`
2. Replace hyphens with spaces: `beware-of-ice` → `beware of ice`
3. Lowercase for consistency (except REACT class names which use exact format)
4. Preserve original case for REACT datasets (e.g., "Abyssinian", "american bulldog")

**Implementation**: [elevater_datasets.py:get_text_templates()](../maveric/datasets/elevater_datasets.py)

---

### Step 3: CLIP Embedding Generation

**Purpose**: Convert reference images and text templates into 512-dimensional CLIP embeddings for similarity matching.

#### Image Embeddings

**Process**:
1. Load CLIP model (default: ViT-B/32)
2. For each reference image:
   - Apply CLIP preprocessing:
     - Resize shortest edge to 224px (preserves aspect ratio)
     - Center crop to 224×224
     - Normalize using CLIP's mean/std
   - Pass through CLIP vision encoder
   - Extract 512-dimensional embedding
3. Store embeddings in memory for processing

**Example**:
```python
# Reference image preprocessing (CORRECT - Nov 2025 fix)
processor(images=image, return_tensors="pt")
# Uses default behavior: resize shortest edge → center crop

# INCORRECT (old approach - caused 6% accuracy drop):
# processor(images=image, size={"height": 224, "width": 224})
# This distorted aspect ratios before cropping
```

**Output Shape**: `[num_images, 512]` (e.g., 100 images × 512 dimensions for CIFAR-10)

#### Text Embeddings

**Process**:
1. For each class and template combination:
   - Format template with class name: `"a photo of a airplane"` (CIFAR-10)
   - Tokenize using CLIP tokenizer
   - Pass through CLIP text encoder
   - Extract 512-dimensional embedding
2. Average embeddings across multiple templates per class (if multiple templates)

**Example**:
```python
# Class: "airplane"
# Templates: ["a photo of a {}.", "a picture of a {}."]
# Generated prompts:
#   - "a photo of a airplane."
#   - "a picture of a airplane."
# → Average embeddings → Final 512-d vector
```

**Output Shape**: `[num_classes, 512]` (e.g., 10 classes × 512 dimensions for CIFAR-10)

**Implementation**: [retriever.py:_generate_reference_embeddings()](../maveric/retrieval/retriever.py)

---

### Step 4: Cache Storage

**Purpose**: Save generated references to disk for reuse across multiple retrieval runs.

#### Reference Images Cache

**Location**: `{cache_base_dir}/reference_images/{dataset_name}/{class_name}/ref_*.jpg`

**Format**: JPEG images (compressed for storage efficiency)

**Example Structure**:
```
maveric_cache/
└── reference_images/
    ├── cifar10/
    │   ├── airplane/
    │   │   ├── ref_000.jpg
    │   │   ├── ref_001.jpg
    │   │   └── ... (10 images)
    │   ├── automobile/
    │   │   └── ... (10 images)
    │   └── ... (10 classes)
    ├── oxford_pets/
    │   ├── Abyssinian/
    │   │   └── ... (10 images)
    │   └── ... (37 classes)
    └── eurosat/
        ├── AnnualCrop/
        │   └── ... (10 images)
        └── ... (10 classes)
```

**Storage Size**: ~500KB - 5MB per dataset (depends on image resolution)

#### Reference Texts Cache

**Location**: `{cache_base_dir}/reference_texts/{dataset_name}_texts.json`

**Format**: JSON file containing templates, class names, and generated prompts

**Schema**:
```json
{
  "templates": [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}."
  ],
  "class_names": [
    "airplane",
    "automobile",
    "bird",
    ...
  ],
  "generated_prompts": {
    "airplane": [
      "a photo of a airplane.",
      "a picture of a airplane.",
      "an image of a airplane."
    ],
    "automobile": [
      "a photo of a automobile.",
      "a picture of a automobile.",
      "an image of a automobile."
    ],
    ...
  }
}
```

**Storage Size**: ~5-50KB per dataset (depends on number of classes and templates)

#### Reference Embeddings Cache

**Location**: `{cache_base_dir}/embeddings/{dataset_name}_reference_embeddings.npz`

**Format**: NumPy compressed array (`.npz` format)

**Contents**:
```python
{
  'image_embeddings': np.ndarray,  # Shape: [n_images, 512]
  'text_embeddings': np.ndarray,   # Shape: [n_classes, 512]
  'class_names': np.ndarray,       # Shape: [n_classes]
  'image_class_mapping': np.ndarray  # Shape: [n_images] - maps each image to class index
}
```

**Storage Size**: ~200KB - 2MB per dataset (depends on number of images/classes)

**Implementation**: [cache_manager.py](../maveric/retrieval/cache_manager.py)

---

## Usage in Retrieval

Once references are generated and cached, they're used during data retrieval as follows:

### 4-Way Similarity Scoring

For each source dataset sample (image + caption), MAVERIC computes similarity with references:

**1. Image-to-Image Similarity** (`img2img`):
```python
# Source image embedding: [1, 512]
# Reference image embeddings: [n_ref_images, 512]
img2img_sim = cosine_similarity(source_img_embed, ref_img_embeds)
# Output: [n_ref_images] similarity scores
# Per-class score: max similarity to any reference image in that class
```

**2. Text-to-Text Similarity** (`txt2txt`):
```python
# Source caption embedding: [1, 512]
# Reference text embeddings: [n_classes, 512]
txt2txt_sim = cosine_similarity(source_txt_embed, ref_txt_embeds)
# Output: [n_classes] similarity scores
```

**3. Image-to-Text Similarity** (`img2txt`):
```python
# Source image embedding: [1, 512]
# Reference text embeddings: [n_classes, 512]
img2txt_sim = cosine_similarity(source_img_embed, ref_txt_embeds)
# Output: [n_classes] similarity scores
```

**4. Text-to-Image Similarity** (`txt2img`):
```python
# Source caption embedding: [1, 512]
# Reference image embeddings: [n_ref_images, 512]
txt2img_sim = cosine_similarity(source_txt_embed, ref_img_embeds)
# Output: [n_ref_images] similarity scores
# Per-class score: max similarity to any reference image in that class
```

### Weighted Class Score

Final similarity score combines all four modalities:

```python
weighted_class_score = (
    0.4 × img2img_similarity +
    0.2 × txt2txt_similarity +
    0.2 × img2txt_similarity +
    0.2 × txt2img_similarity
)
```

**Per-Class Scores**: Each target class gets its own weighted score, stored as:
- `Class_airplane_img2img`
- `Class_airplane_txt2txt`
- `Class_airplane_img2txt`
- `Class_airplane_txt2img`
- `Class_airplane_weighted_class_score`

**Metric Weights Configuration**:
```yaml
metric_weights:
  img2img: 0.4  # Image-to-image similarity (highest weight)
  txt2txt: 0.2  # Text-to-text similarity
  img2txt: 0.2  # Image-to-text cross-modal
  txt2img: 0.2  # Text-to-image cross-modal
```

---

## Usage in Model Customization

References are also used during model fine-tuning:

### Training Data Preparation

**Class Name Consistency**:
- Training JSON may have lowercase/normalized labels: `"abyssinian"`, `"speed_limit_30"`
- Evaluation uses REACT class names: `"Abyssinian"`, `"speed limit 30"`
- **Solution**: Case-insensitive label mapping during training ([model_customizer.py:847-850](../maveric/customization/model_customizer.py#L847-L850))

**Example**:
```python
# Training data: {"label": "abyssinian", "image": ...}
# REACT class names: ["Abyssinian", "american bulldog", ...]

# Create normalized mapping:
label_map = {}
for idx, class_name in enumerate(class_names):
    label_map[class_name.lower()] = idx  # "abyssinian" → 0
    label_map[class_name] = idx          # "Abyssinian" → 0

# Works for both:
class_idx = label_map["abyssinian"]  # → 0
class_idx = label_map["Abyssinian"]  # → 0
```

### Test Dataset Creation

**REACT Template Usage**:
- Load EXACT class names from `ELEVATER_DATASETS` dictionary
- Apply dataset-specific REACT templates
- Generate test samples with proper formatting

**Example** (Oxford Pets):
```python
# Class names from ELEVATER_DATASETS (REACT format):
class_names = ["Abyssinian", "american bulldog", "Bengal", ...]

# Templates:
templates = [
    "a photo of a {}, a type of pet.",
    "a photo of a {}.",
    "a {} looking at the camera."
]

# Test samples:
# Class: "Abyssinian"
# → "a photo of a Abyssinian, a type of pet."
# → "a photo of a Abyssinian."
# → "a Abyssinian looking at the camera."
```

**Critical Fix (Nov 2025)**:
- Load class names from `ELEVATER_DATASETS`, NOT from dataset handler
- Avoids torchvision overriding with dynamically-generated class names
- Ensures EXACT REACT class names throughout pipeline
- [03_model_customization.py:331-363](../experiments/03_model_customization.py#L331-L363)

### Evaluation with References

**Template Application**:
```python
# During evaluation:
text_inputs = []
for class_name in class_names:
    # Get dataset-specific templates
    templates = dataset.get_text_templates()

    # Generate prompts for this class
    for template in templates:
        prompt = template.format(class_name)
        text_inputs.append(prompt)

# Example (EuroSAT, class "Forest"):
# → "a centered satellite photo of Forest."
# → "a satellite photo of Forest."
```

**Accuracy Impact**: Using correct REACT templates improves accuracy by ~4-5% compared to generic templates.

---

## Performance Benefits

### Cache Reuse Across Datasets

**Scenario**: Retrieve data for all 20 ELEVATER datasets

**Without Caching** (regenerate references each time):
- 20 datasets × 2.2 hours = **44.4 hours total**

**With Caching** (generate once, reuse):
- First run: 2.2 hours (generate + cache)
- Next 19 runs: 19 × 0.3 hours = 5.7 hours
- **Total: 7.9 hours (82% time savings)**

### Cross-Dataset Sample Caching (v3)

Reference embeddings integrate with sample metadata caching:

**Sample Cache v3 Contents** (per source sample):
- Visual metrics (resolution, sharpness, color_diversity)
- Semantic metrics (text_quality, caption_length)
- **CLIP embeddings** (image + text, 512-d each)
- EfficientNet predictions (ImageNet class + probability)

**Storage Trade-off**:
- **v3 (with CLIP embeddings)**: ~17KB per sample = 4.5GB for 270K samples
- **v2 (without embeddings)**: ~500 bytes per sample = 135MB for 270K samples
- **Benefit**: Eliminates CLIP inference on cache hits (saves 150-700ms per sample)

**Cache Hit Performance**:
```
Cache v2 (compute CLIP from cached images):
  - Load image from cache: ~50ms
  - CLIP inference: ~150-700ms
  - Total: ~200-750ms per sample

Cache v3 (load CLIP embeddings directly):
  - Load embeddings from JSON: ~5-10ms
  - Total: ~5-10ms per sample (95%+ faster)
```

**Configuration**:
```yaml
enable_sample_cache: true
sample_cache_version: 3  # Use v3 for CLIP embedding caching
```

### Reference Embedding Size

**Example** (CIFAR-10):
- 10 classes
- 10 reference images per class = 100 images total
- Image embeddings: 100 × 512 × 4 bytes = 204KB
- Text embeddings: 10 × 512 × 4 bytes = 20KB
- Metadata: ~5KB
- **Total**: ~230KB (negligible)

**Example** (Food101):
- 101 classes
- 10 reference images per class = 1,010 images total
- Image embeddings: 1,010 × 512 × 4 bytes = 2.06MB
- Text embeddings: 101 × 512 × 4 bytes = 206KB
- Metadata: ~20KB
- **Total**: ~2.3MB (still very small)

---

## Best Practices

### 1. Reference Image Selection

**Recommended**:
- Use 10-20 reference images per class (default: 10)
- Ensure diversity within each class
- Use clean, representative samples

**Not Recommended**:
- Too few references (< 5) → Poor similarity matching
- Too many references (> 50) → Diminishing returns, slower processing

### 2. Template Design

**Good Templates**:
- Contextually appropriate: "satellite photo" for EuroSAT
- Domain-specific: "traffic sign" for GTSRB
- Natural language: "a photo of a {} texture" (DTD)

**Poor Templates**:
- Too generic: "an image" (loses context)
- Unnatural: "photo airplane" (grammatically incorrect)
- Inconsistent: Mixing styles within same dataset

### 3. Cache Management

**Enable Caching** (always recommended):
```yaml
enable_image_cache: true
enable_sample_cache: true
sample_cache_version: 3  # Use latest version
```

**Clear Cache When**:
- Changing CLIP model version
- Updating text templates
- Modifying reference image selection
- Upgrading cache version

**Clear Cache Command**:
```python
from maveric import MAVERIC
maveric = MAVERIC()
maveric.cache_manager.clear_cache()  # Clear all caches

# Or clear specific dataset:
maveric.cache_manager.clear_cache(dataset_name="cifar10")
```

### 4. REACT Template Consistency

**Critical**:
- Always use EXACT class names from `ELEVATER_DATASETS`
- Never modify class names between retrieval and evaluation
- Use same templates for training and evaluation

**Example Pitfall**:
```python
# WRONG:
class_names = test_dataset.class_names  # May be modified by torchvision
# e.g., Oxford Pets: ["American Bulldog", ...] (Title Case)

# CORRECT:
from maveric.datasets.elevater_datasets import ELEVATER_DATASETS
class_names = ELEVATER_DATASETS["oxford_pets"]["class_names"]
# e.g., ["american bulldog", ...] (REACT's mixed-case format)
```

---

## Troubleshooting

### Issue: Reference generation is slow

**Symptoms**: First-time reference generation takes > 5 minutes

**Causes**:
1. Large dataset (e.g., Food101 with 75K training samples)
2. Slow storage (Google Drive over network)
3. CLIP model loading time

**Solutions**:
- Expected for large datasets (progress logging shows status)
- Use local SSD instead of network storage when possible
- First-time setup is one-time cost; subsequent runs use cache

### Issue: Template formatting errors

**Symptoms**: Generated prompts look incorrect (e.g., "a photo of a speed_limit_30")

**Cause**: Class name normalization not applied

**Solution**: Verify `get_text_templates()` properly formats class names
```python
# Check generated prompts:
dataset = ELEVATER_DATASETS["gtsrb"]
templates = dataset.get_text_templates()
class_name = "speed_limit_30"
formatted_name = class_name.replace("_", " ")  # "speed limit 30"
prompt = templates[0].format(formatted_name)
# → "a zoomed in photo of a speed limit 30 traffic sign"
```

### Issue: Class name mismatch errors

**Symptoms**: "Class 'Abyssinian' not found" or accuracy drops significantly

**Cause**: Using torchvision's dynamically-generated class names instead of REACT names

**Solution**: Load class names from `ELEVATER_DATASETS`:
```python
# WRONG:
from torchvision.datasets import OxfordIIITPet
dataset = OxfordIIITPet(root="./data", split="test")
class_names = dataset.classes  # Dynamic Title Case names

# CORRECT:
from maveric.datasets.elevater_datasets import ELEVATER_DATASETS
class_names = ELEVATER_DATASETS["oxford_pets"]["class_names"]
# Exact REACT mixed-case format
```

### Issue: Reference embeddings not loading from cache

**Symptoms**: References regenerated on every run despite cache existing

**Causes**:
1. Cache file corrupted (numpy scalar array issue - fixed Nov 2025)
2. Different CLIP model version
3. Cache directory permissions

**Solutions**:
- Check cache file exists: `{cache_base_dir}/embeddings/{dataset}_reference_embeddings.npz`
- Verify CLIP model matches: Model name in config vs. cached embeddings
- Check file permissions: Ensure read/write access to cache directory

**Cache Validation Fix** (Nov 2025):
```python
# OLD (failed validation):
ref_cache = np.load(cache_path)
if isinstance(ref_cache, dict):  # FAILS for numpy scalar arrays
    return ref_cache

# NEW (handles scalar arrays):
ref_cache = np.load(cache_path)
if isinstance(ref_cache, np.ndarray) and ref_cache.ndim == 0:
    ref_cache = ref_cache.item()  # Extract dict from scalar array
if isinstance(ref_cache, dict):
    return ref_cache
```

---

## Summary

Reference generation is a foundational component of MAVERIC that:

1. **Generates once, uses many times**: One-time setup per dataset with persistent caching
2. **Enables accurate retrieval**: 4-way similarity matching using reference embeddings
3. **Improves model performance**: REACT-style templates boost evaluation accuracy by 4-5%
4. **Optimizes performance**: 80-95% speedup through embedding caching
5. **Ensures consistency**: Same references used for retrieval and evaluation

**Key Takeaways**:
- Always use EXACT class names from `ELEVATER_DATASETS`
- Enable sample cache v3 for maximum performance
- Use dataset-specific REACT templates for best accuracy
- Trust cached references unless changing fundamental parameters

---

## Related Documentation

- [Cache Management Guide](CROSS_DATASET_CACHING.md)
- [ELEVATER Datasets Reference](../maveric/datasets/elevater_datasets.py)
- [Retrieval System Architecture](../CLAUDE.md#retrieval-system-mavericretrieval)
- [Model Customization Guide](../CLAUDE.md#customization-system-mavericcustomization)
