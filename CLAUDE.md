# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference - Recent Updates

### July 18, 2026 - VOC2007 Multi-Label Evaluation Fix (CRITICAL FIX)

**Bug Fix: VOC2007 mAP Evaluation Now Matches ELEVATER Baseline**:
- **Problem**: VOC2007 baseline showed 69.99% mAP instead of expected 82.60%
- **Root causes**:
  1. Multi-label dataset loaded as single-label via ImageFolder (destroyed multi-label structure)
  2. Double-scaling in softmax normalization: `softmax(scores * 100)` when CLIP already applies 100x internally
- **Solution**: 
  - Created `VOC2007MultiLabelDataset` reading ImageSets/Main annotations for proper multi-hot labels
  - Fixed softmax: `softmax(scores)` NOT `softmax(scores * 100)` (CLIP model already scales by 100x)
  - Implemented 11-point interpolated mAP per VOC protocol
- **Results**: 69.99% → **82.57%** (+12.58 points) ✅
  - Multi-label dataset fix: +13.71 points (61.36% → 75.07%)
  - Softmax correction: +7.50 points (75.07% → 82.57%)
  - **Match with ELEVATER**: 82.57% vs 82.60% expected (within 0.03%)
- **Files Modified**:
  - [elevater_datasets.py](maveric/datasets/elevater_datasets.py#L17-L145) - VOC2007MultiLabelDataset class
  - [model_customizer.py](maveric/customization/model_customizer.py#L386-L477) - _create_voc2007_test_loader()
  - [evaluation.py](maveric/customization/evaluation.py#L531-L559) - evaluate_with_dataset_metric() with voc11_map
  - [evaluation.py](maveric/customization/evaluation.py#L376-L420) - _compute_voc11_map() implementation
- **Key Insight**: CLIP model internally applies 100x temperature scaling (`logits = 100.0 * image_embeds @ text_features.T`). ELEVATER's `softmax(100. * features)` means softmax of already-scaled logits, NOT an additional 100x multiplication.
- **Documentation**:
  - [VOC2007_FIX_SUMMARY.md](docs/bugfixes/VOC2007_FIX_SUMMARY.md) - Complete debugging timeline and solution
  - [ELEVATER_MATCHING_IMPLEMENTATION.md](docs/bugfixes/ELEVATER_MATCHING_IMPLEMENTATION.md) - ELEVATER matching details
  - [MAP_IMPLEMENTATION_COMPARISON.md](docs/bugfixes/MAP_IMPLEMENTATION_COMPARISON.md) - Our mAP vs ELEVATER's sklearn-based mAP
- **Testing**: All 20 ELEVATER datasets now use correct per-dataset metrics (accuracy, mean_per_class, roc_auc, voc11_map)

### July 6, 2026 - Grid Visualization Performance Optimization (NEW)

**Enhancement: Optional Grid Visualization for Better Performance**:
- **Purpose**: Make 10×10 image grid generation optional to speed up data curation
- **Problem**: Grid visualization takes significant time for large datasets (e.g., Food101)
- **Solution**: New `save_grid_visualization` config parameter (disabled by default)
- **Configuration**:
  ```yaml
  # maveric_config.yaml (top-level, after visualization section)
  save_grid_visualization: false  # Default: disabled for performance
  ```
- **Files Modified**:
  - [config.py](maveric/config.py#L100) - Added `save_grid_visualization` parameter
  - [interactive.py](maveric/visualization/interactive.py#L109) - Instance variable initialization
  - [interactive.py](maveric/visualization/interactive.py#L195-L198) - Config loading
  - [interactive.py](maveric/visualization/interactive.py#L3356-L3366) - Conditional grid generation
  - [maveric_config.yaml](experiments/maveric_config.yaml#L285-L287) - Config documentation
- **Behavior**:
  - **Default (false)**: Clicking "Save Data" skips grid generation, shows info message
  - **Enabled (true)**: Generates 10×10 grids as before (December 2025 feature)
  - User informed when grids are skipped with instructions to enable
- **Benefits**:
  - **Faster curation**: No grid generation overhead by default
  - **User control**: Enable only when visual inspection needed
  - **Backward compatible**: Feature still available, just opt-in

### June 21, 2026 - Caption-Based Training Mode (NEW FEATURE)

**Enhancement: Alternative Text Source for CLIP Fine-Tuning**:
- **Purpose**: Allow switching between class labels and original image captions during model customization
- **Configuration**: New `text_source` parameter in `TrainingConfig` with two modes:
  - `"labels"` (default): Use class names converted to prompts (e.g., "a photo of a airplane")
  - `"captions"`: Use original caption from `text` field (e.g., "a red airplane flying in the sky")
- **Implementation**: Config-driven mode selection with InfoNCE contrastive loss for caption mode
- **Key Benefits**:
  - Zero breaking changes - default behavior unchanged
  - Backward compatible - existing configs work without modification
  - Richer semantic information in caption mode
  - Single config parameter switches entire training pipeline
- **Files Added**:
  - [losses.py](maveric/customization/losses.py) - InfoNCELoss for contrastive learning
- **Files Modified**:
  - [config.py](maveric/config.py#L323) - Added `text_source` parameter to TrainingConfig
  - [maveric_config.yaml](experiments/maveric_config.yaml#L203-L204) - Added text_source configuration option
  - [03_model_customization.py](experiments/03_model_customization.py#L561) - Extract text_source from config
  - [model_customizer.py](maveric/customization/model_customizer.py#L1067) - text_source parameter in LAIONCustomDataset
  - [model_customizer.py](maveric/customization/model_customizer.py#L270) - Wire text_source through _prepare_data
  - [training.py](maveric/customization/training.py#L76-L115) - Caption mode detection and InfoNCE loss
  - [training.py](maveric/customization/training.py#L251-L315) - Modified _train_epoch for both modes
  - [training.py](maveric/customization/training.py#L318-L366) - Added _forward_caption_mode method

**Training Mode Characteristics**:
- **Label Mode** (default, `text_source: "labels"`):
  - Pre-computes text features for all classes using templates
  - Uses CrossEntropyLoss for classification
  - Text encoder remains frozen (locked-text tuning)
  - Consistent with zero-shot CLIP evaluation
  - Best for classification tasks with clean class names

- **Caption Mode** (`text_source: "captions"`):
  - Computes text features per-batch from sample captions
  - Uses InfoNCELoss for contrastive learning (CLIP-style)
  - Text encoder remains frozen (locked-text tuning)
  - Learns from natural language descriptions
  - Best for fine-grained visual understanding
  - Expected ~20-30% slower due to per-batch text encoding

**Usage**:
```yaml
# maveric_config.yaml
training:
  text_source: "labels"    # Default: label-based prompts
  # OR
  text_source: "captions"  # Alternative: caption-based contrastive learning
```

**Important Notes**:
- Validation and test evaluation always use label-based evaluation for consistency
- Text encoder remains frozen in both modes (only vision encoder is fine-tuned)
- Caption mode uses symmetric InfoNCE loss (image-to-text + text-to-image)
- Training data must have valid `text` field for caption mode
- No changes needed to existing datasets or workflows

### March 26, 2026 - Hu Moments Alternative Scoring Mode (NEW FEATURE)

**Enhancement: Shape-Based Similarity Scoring via Hu Invariant Moments**:
- **Purpose**: Alternative to CLIP-based multi-modal scoring using Hu invariant moments for shape-based image similarity
- **Motivation**: Based on Wu et al. (2020) showing 12.5%-56.25% per-class retrieval accuracy on CIFAR-10, with shape-distinctive classes performing better
- **Implementation**: Config-driven scoring mode selection (`scoring_mode: "clip"` or `"hu_moments"`)
- **Key Benefits**:
  - Zero breaking changes - all existing CLIP functionality preserved
  - Single config parameter switches entire pipeline
  - Auto-detection of mode from column names (no manual specification needed)
  - Same curated output format for both modes
- **Files Added**:
  - [hu_moments_metric.py](maveric/quality/metrics/hu_moments_metric.py) - HuMomentsSimilarityMetric class
- **Files Modified**:
  - [config.py](maveric/config.py#L39) - Added `scoring_mode` parameter with validation
  - [retriever.py](maveric/retrieval/retriever.py#L51-L56) - Hu reference vectors and branched similarity computation
  - [quality_controller.py](maveric/quality/quality_controller.py#L117-L225) - Auto-detection and mode-specific best class calculation
  - [interactive.py](maveric/visualization/interactive.py#L274-L400) - GUI support with auto-detection
  - [main.py](maveric/main.py#L88) - Wired scoring_mode to Retriever
  - [sample_cache_manager.py](maveric/retrieval/sample_cache_manager.py#L218-L283) - Optional Hu vector caching
  - [maveric_config.yaml](experiments/maveric_config.yaml#L36) - Added scoring_mode configuration option

**Hu Moments Mode Characteristics**:
- **Metric**: `Class_{class_name}_hu_similarity` (shape-based, rotation/scale/translation invariant)
- **Algorithm**: First 2 Hu invariant moments (h1, h2) → log transform → Euclidean distance → `1/(1+d)` similarity
  - h1: Area-related feature
  - h2: Aspect ratio feature
  - Higher moments (h3-h7) excluded for better stability and discriminability
- **Consistency**: Always 1.0 (no cross-modal agreement in shape-only mode)
- **Visual Metrics**: Still computed (resolution, sharpness, color) regardless of mode
- **Reference**: 10 reference images per class (same as CLIP mode)

**CLIP Mode Characteristics** (default, unchanged):
- **Metrics**: `img2img`, `txt2txt`, `img2txt`, `txt2img`, `hybrid_score`, `consistency`
- **Algorithm**: CLIP embeddings → cosine similarity → weighted average
- **Reference**: 10 reference images + 18 text templates per class

**Output Format Comparison**:
```python
# CLIP mode (default):
{
  "Class_airplane_img2img": 0.234,
  "Class_airplane_txt2txt": 0.189,
  "Class_airplane_img2txt": 0.201,
  "Class_airplane_txt2img": 0.178,
  "Class_airplane_consistency": 0.823,
  "resolution_score": 2.143,
  "sharpness_score": 0.891,
  "color_score": 0.823
}

# Hu moments mode:
{
  "Class_airplane_hu_similarity": 0.567,
  "resolution_score": 2.143,
  "sharpness_score": 0.891,
  "color_score": 0.823
}

# Both produce identical curated output:
{
  "label": "truck",
  "weighted_class_score": 0.567,
  "consistency": 1.0,  # Always 1.0 in Hu mode
  "url": "https://...",
  "text": "a red truck"
}
```

**Usage**:
```yaml
# maveric_config.yaml
scoring_mode: "clip"       # Default - multi-modal CLIP-based
scoring_mode: "hu_moments" # Alternative - shape-based Hu moments
```

**Reference**: Wu et al., "Application of image retrieval based on CNN and Hu invariant moment algorithm," Computer Communications, 2020

### February 10, 2026 - Dataset Type Migrations (BREAKING CHANGES)

**Major Changes: Two Datasets Migrated from Torchvision to File-Based**:

#### Food101 Migration to File-Based
- **Type Change**: 'torchvision' → 'file_based'
- **Class Names**: All 101 names changed from spaces to underscores
  - Examples: 'apple pie' → 'apple_pie', 'chocolate cake' → 'chocolate_cake', 'baby back ribs' → 'baby_back_ribs'
- **Impact**: No automatic download - requires manual test data setup
- **Setup Guide**: See [FILE_BASED_DATASETS_GUIDE.md](FILE_BASED_DATASETS_GUIDE.md) for Food101 download instructions
- **Location**: [elevater_datasets.py:588-697](maveric/datasets/elevater_datasets.py#L588-L697)
- **Commit**: 4f75f38 "Change food101 type from 'torchvision' to 'file_based'"

#### FGVCAircraft Migration to File-Based
- **Type Change**: 'torchvision' → 'file_based'
- **Class Names**: 27 aircraft model names changed from spaces to underscores
  - Examples: 'BAE 146-200' → 'BAE_146-200', 'Boeing 717' → 'Boeing_717', 'Beechcraft 1900' → 'Beechcraft_1900'
- **Impact**: No automatic download - requires manual test data setup
- **Setup Guide**: See [FILE_BASED_DATASETS_GUIDE.md](FILE_BASED_DATASETS_GUIDE.md) for FGVCAircraft download instructions
- **Location**: [elevater_datasets.py:481-587](maveric/datasets/elevater_datasets.py#L481-L587)
- **Commit**: e041312 "Refactor aircraft names for consistency"

#### Updated Dataset Counts
- **Torchvision datasets**: ~~9~~ → **7** (removed Food101, FGVCAircraft)
  - CIFAR-10, CIFAR-100, Country211, EuroSAT, GTSRB, Oxford Flowers102, Oxford Pets
- **File-based datasets**: ~~8~~ → **13** (added Food101, FGVCAircraft, plus DTD, MNIST, Caltech101)
  - Caltech101, DTD, FER2013, FGVCAircraft, Food101, HatefulMemes, Kitti Distance, MNIST, PatchCamelyon, RenderedSST2, RESISC45, StanfordCars, VOC2007
- **Total ELEVATER datasets**: Still 20 (no change)

### February 10, 2026 - Caltech101 Class Count Update (REVERSAL)

**Change: Reverted to 102 Classes (with 'background_google')**:
- **Previous State** (November 19, 2025): 101 classes (excluded 'BACKGROUND_Google' to match torchvision)
- **New State**: 102 classes (includes 'background_google' as lowercase)
- **Class Ordering**: 'faces' and 'faces_easy' reordered alphabetically
- **Impact**: Reverses November 2025 fix that removed BACKGROUND_Google for torchvision compatibility
- **Note**: This may reintroduce index mismatch issues with torchvision (which excludes BACKGROUND_Google)
- **Location**: [elevater_datasets.py:30](maveric/datasets/elevater_datasets.py#L30)
- **Commit**: 85986d0 "Update number of classes for caltech101 dataset"
- **Clarification**: Torchvision compatibility implications need verification

### February 8, 2026 - Corrupt Image Handling

**Enhancement: Automatic Placeholder for Degenerate Images**:
- **Problem**: Corrupt or extremely small images (<10×10 pixels) caused processing failures
- **Solution**: Automatically replace degenerate images with 224×224 gray placeholder
- **Threshold**: Images with width OR height < 10 pixels
- **Placeholder**: RGB(128, 128, 128) gray image at 224×224 resolution
- **Location**: [model_customizer.py:879-884](maveric/customization/model_customizer.py#L879-L884)
- **Benefit**: Prevents training crashes from corrupt/unusual images in datasets
- **Commit**: f2c707c "A corrupt/unusual image caused manual processing, replace them with a placeholder image"

### February 8, 2026 - VOC2007 Multi-Label Evaluation (ATTEMPTED - REVERTED)

**Note: Multi-label evaluation was attempted but fully reverted**:
- **Attempt**: Implemented comprehensive mAP evaluation for VOC2007 (inherently multi-label dataset)
- **Scope**: Added multi-label support across 5 files (evaluation.py, model_customizer.py, training.py, elevater_datasets.py, CLAUDE.md)
- **Status**: **Fully reverted after 57 minutes** - reason unknown
- **Current State**: VOC2007 remains single-label classification (may not match REACT benchmark expectations)
- **Commits**:
  - Implementation: 2e6449c "VOC2007 Multi-Label Evaluation Fix" (February 8, 12:20)
  - Revert: d5203b0 "Revert VOC2007 Multi-Label Evaluation Fix" (February 8, 13:17)
- **Impact**: VOC2007 baseline accuracy may differ from REACT benchmark
- **Future Consideration**: Multi-label evaluation may need to be re-implemented with different approach

### February 6, 2026 - Transformers Version Pinning (RECOMMENDED)

**Decision: Pin Transformers to <5.0.0 for Maximum Stability**:
- **Change**: Updated `requirements.txt` to specify `transformers>=4.20.0,<5.0.0`
- **Rationale**: Prevent automatic upgrades to transformers 5.0.0+ which introduced breaking changes
- **Benefits**:
  - Maximum stability with known working configuration
  - Avoids large safetensors downloads (uses pytorch_model.bin format)
  - Predictable behavior for reproducible research
  - Prevents future breaking changes without warning
- **Documentation**:
  - [TRANSFORMERS_VERSION_GUIDE.md](TRANSFORMERS_VERSION_GUIDE.md) - Complete guide with 3 solution options
  - [QUICK_ROLLBACK_GUIDE.md](QUICK_ROLLBACK_GUIDE.md) - One-command rollback solutions
- **Installation**: `pip install -r requirements.txt` now pins to transformers 4.x
- **Note**: Code remains backward compatible with both 4.x and 5.x versions (fix from February 4 still active)
- **Location**: [requirements.txt:4-6](requirements.txt#L4-L6)
- **Commit**: ffa86f8 "Rollingback to transformers library version before 5.0.0"

### February 4, 2026 - Caltech101 Class Name Case Corrections

**Change: Mixed-Case to Lowercase for Consistency**:
- **Previous State** (January 30): Mixed-case ('Faces_easy', 'Leopards', 'Motorbikes') documented as "intentional"
- **New State**: All lowercase for consistency ('faces_easy', 'leopards', 'motorbikes')
- **Changes**:
  - 'Faces_easy' → 'faces_easy'
  - 'Faces' → 'faces'
  - 'Leopards' → 'leopards'
  - 'Motorbikes' → 'motorbikes'
- **Note**: Reverses January 30 mixed-case standardization after only 5 days
- **Location**: [elevater_datasets.py:67-100](maveric/datasets/elevater_datasets.py#L67-L100)
- **Commit**: 7bad6cd "Caltech101 class name corrections"

### February 4, 2026 - HuggingFace Transformers API Compatibility Fix (CRITICAL)

**Bug Fix: BaseModelOutputWithPooling Compatibility**:
- **Problem**: HuggingFace transformers library updated CLIP model format from `pytorch_model.bin` to `model.safetensors` (via PR #66), changing `get_text_features()` return type from `Tensor` to `BaseModelOutputWithPooling` object
- **Impact**: Model customization crashed with `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'` and later `RuntimeError: stack expects each tensor to be equal size` due to incorrect tensor extraction
- **Root Cause**: Code assumed `get_text_features()` returns a plain tensor, but new safetensors format returns a wrapped object. Initial fix used `[0]` which extracts `last_hidden_state` (shape: batch_size, seq_len, hidden_size) instead of `pooler_output` (shape: batch_size, hidden_size)
- **Fix** (January 29, 2026): Corrected to use `pooler_output` attribute for proper tensor extraction:
  ```python
  text_features_output = model.clip_model.get_text_features(**text_inputs)

  # Handle both tensor and BaseModelOutputWithPooling formats
  if isinstance(text_features_output, torch.Tensor):
      text_embeds = text_features_output
  else:
      # Extract pooler_output from BaseModelOutputWithPooling (shape: batch_size, hidden_size)
      # NOT [0] which gives last_hidden_state (shape: batch_size, seq_len, hidden_size)
      text_embeds = text_features_output.pooler_output if hasattr(text_features_output, 'pooler_output') else text_features_output[0]
  ```
- **Locations Fixed** (January 29, 2026 - corrected to use `pooler_output`):
  - [evaluation.py:94-101](maveric/customization/evaluation.py#L94-L101) - `_create_text_classifier_with_templates()` method (text features)
  - [evaluation.py:155-160](maveric/customization/evaluation.py#L155-L160) - `evaluate()` method (text features)
  - [evaluation.py:208-213](maveric/customization/evaluation.py#L208-L213) - `evaluate_detailed()` method (text features)
  - [training.py:88-93](maveric/customization/training.py#L88-L93) - Training loop text features
  - [model_customizer.py:795-803](maveric/customization/model_customizer.py#L795-L803) - `forward()` method (image features)
  - [model_customizer.py:971-976](maveric/customization/model_customizer.py#L971-L976) - `encode_text()` method
- **Testing**: Verified with Caltech101 model customization (6,084 test samples)
- **Backward Compatibility**: Maintains support for both old (`pytorch_model.bin`) and new (`model.safetensors`) formats
- **Note**: This is a breaking change from HuggingFace, not MAVERIC. The fix ensures MAVERIC works with both old and new transformers versions.

### January 30, 2026 - Caltech101 Class Name Standardization (LATER CORRECTED)

**Refactoring: Consistent Class Naming for Caltech101 Dataset**:
- **Purpose**: Standardize Caltech101 class names for consistency with torchvision's implementation
- **Changes**: 48 class names updated in ELEVATER_DATASETS dictionary
- **Naming Convention**:
  - Use underscores instead of spaces or descriptive phrases
  - Maintain consistency with torchvision's Caltech101 class names
  - ~~Some classes retain capitalization to match torchvision format exactly~~ (Corrected to lowercase on February 4)
- **Examples of Changes**:
  - 'airplane' → 'airplanes' (pluralization to match torchvision)
  - 'side of a car' → 'car_side' (standardized format)
  - 'ceiling fan' → 'ceiling_fan' (underscore instead of space)
  - 'body of a cougar cat' → 'cougar_body' (concise naming)
  - 'centered face' → ~~'Faces_easy'~~ → 'faces_easy' (February 4 correction)
  - 'off-center face' → ~~'Faces'~~ → 'faces' (February 4 correction)
  - 'leopard' → ~~'Leopards'~~ → 'leopards' (February 4 correction)
  - 'motorbike' → ~~'Motorbikes'~~ → 'motorbikes' (February 4 correction)
  - 'dollar bill' → 'dollar_bill' (underscore format)
- **Location**: [elevater_datasets.py:33-128](maveric/datasets/elevater_datasets.py#L33-L128)
- **Impact**:
  - Affects all Caltech101 dataset users
  - Class names now match torchvision's exact format
  - Improves consistency across MAVERIC's dataset handling
  - Resolves ambiguities in class naming conventions
- **Related Fixes**: Caltech101 improvements timeline:
  - November 18, 2025: Added missing "leopards" class
  - November 19, 2025: Fixed class list mismatch (102 → 101 classes, excluding 'BACKGROUND_Google')
  - January 30, 2026: Standardized all class names with mixed-case
  - **February 4, 2026**: Corrected to all lowercase (see separate entry above)
  - **February 10, 2026**: Reverted to 102 classes with 'background_google' (see separate entry above)
- **Note**: ~~Mixed capitalization was initially documented as "intentional"~~ but was corrected to lowercase on February 4, 2026
- **Documentation**: See commit d4c8a69 "Refactor class names for consistency and clarity"

### January 5, 2026 - Mahalanobis Batch Processing "ALL" Feature

**Enhancement: Batch Process All Classes at Once in Class-Based Mode**:
- **Purpose**: Dramatically speed up workflow for datasets with many classes (CIFAR-100, etc.)
- **New Features**:
  - **'ALL' Option**: New option in class selector dropdown for batch processing
  - **Batch Filtering**: Automatically filters all classes with same parameters
  - **Progress Updates**: Shows progress every 10 classes
  - **Summary Table**: Displays filtered sample counts per class
  - **No Plots**: Skips visualization for speed (summary table only)
  - **Error Handling**: Continues processing even if individual classes fail
- **Location**: [interactive.py:1321, 1468, 1557-1635, 1680-1702](maveric/visualization/interactive.py)
- **Benefits**:
  - **10x Faster**: 2-3 minutes instead of 20-30 minutes for CIFAR-100
  - **100x Fewer Clicks**: 5 clicks instead of 500 for 100 classes
  - **Consistent Filtering**: Same parameters applied to all classes
  - **Clear Feedback**: Comprehensive summary table before committing
- **Usage Example**:
  ```python
  # Select 'ALL' from class dropdown
  # Set Keep Count: 350
  # Click "Apply Filter" → Processes all 100 classes
  # Review summary table
  # Click "Add Data" → Done!
  ```
- **Console Output**:
  ```
  🔄 Starting batch processing for 100 classes...
     Progress: 10/100 classes processed (10 successful, 0 failed)
     Progress: 20/100 classes processed (20 successful, 0 failed)
     ...
     Progress: 100/100 classes processed (100 successful, 0 failed)

  ✅ Batch processing complete!
     Successful: 100/100 classes

  📊 Filtered Samples by Class:
     apple                         :    350 /  5,000 ( 7.0%)
     aquarium_fish                 :    350 /  5,000 ( 7.0%)
     ... (98 more classes)
     ────────────────────────────────────────────────────────
     TOTAL                         : 35,000 samples
  ```
- **Documentation**: [MAHALANOBIS_BATCH_ALL_FEATURE.md](MAHALANOBIS_BATCH_ALL_FEATURE.md)

### January 28, 2026 - File-Based Datasets Test Data Issue (LATEST)

**Issue Identified: Test Data Loading for File-Based Datasets**:
- **Problem**: When running `03_model_customization.py`, torchvision datasets (CIFAR-10, GTSRB, etc.) work fine, but file-based datasets fail to load test data for evaluation
- **Root Cause**: File-based datasets don't have automatic download support like torchvision datasets. The `_load_dataset()` method does nothing for file-based datasets (`pass` at line 1285), leaving `self._dataset = None`, which causes test data loading to fail in `_create_test_loader()` at line 359
- **Affected Datasets** (~~8~~ → **13 total** as of February 10, 2026):
  - Caltech101 (Object recognition - 102 classes)
  - DTD (Texture recognition - 47 classes)
  - FER2013 (Facial Expression Recognition - 7 classes)
  - **FGVCAircraft** (Aircraft recognition - 100 classes) ⬅️ **NEW** (migrated February 10)
  - **Food101** (Food recognition - 101 classes) ⬅️ **NEW** (migrated February 10)
  - HatefulMemes (Hate speech detection - 2 classes)
  - Kitti Distance (Car distance estimation - 4 classes)
  - MNIST (Digit recognition - 10 classes)
  - PatchCamelyon/PCAM (Lymph node classification - 2 classes)
  - RenderedSST2 (Sentiment analysis - 2 classes)
  - RESISC45 (Remote sensing - 45 classes)
  - StanfordCars (Car recognition - 196 classes)
  - VOC2007 (Object recognition - 20 classes)
- **Solutions Documented**:
  - **Option 1 (Immediate)**: Manual download + directory organization
    - Download datasets from official sources
    - Organize into expected structure: `{root}/elevater/{dataset_name}/test/class_name/*.jpg`
    - Documented with dataset-specific instructions for all 8 datasets
  - **Option 2 (Best long-term)**: Implement PyTorch Dataset wrappers
    - Create custom dataset classes with automatic download support
    - Change dataset type from 'file_based' to 'torchvision' in ELEVATER_DATASETS
    - Provides consistent interface like existing torchvision datasets
  - **Option 3 (Workaround)**: Use file-based datasets for retrieval only
    - Train on file-based data, evaluate on torchvision datasets instead
    - Allows workflow continuation without test data setup
- **Documentation**: [FILE_BASED_DATASETS_GUIDE.md](FILE_BASED_DATASETS_GUIDE.md) - Complete setup guide with download sources, directory structures, and extraction scripts for all file-based datasets
- **Code Locations**:
  - Dataset type detection: [elevater_datasets.py:1278](maveric/datasets/elevater_datasets.py#L1278)
  - File-based loading (empty): [elevater_datasets.py:1283-1285](maveric/datasets/elevater_datasets.py#L1283-L1285)
  - Test loader creation: [model_customizer.py:339-451](maveric/customization/model_customizer.py#L339-L451)
  - Error handling with setup instructions: [model_customizer.py:435-450](maveric/customization/model_customizer.py#L435-L450)
  - Dataset check that fails: [model_customizer.py:359-360](maveric/customization/model_customizer.py#L359-L360)
- **Expected Directory Structure**:
  ```
  {root}/elevater/{dataset_name}/
  ├── train/              # Training split (for reference samples)
  │   ├── class_1/
  │   │   ├── image_001.jpg
  │   │   └── ...
  │   └── ...
  └── test/               # Test split (for evaluation)
      ├── class_1/
      │   ├── image_001.jpg
      │   └── ...
      └── ...
  ```
- **Critical Fix** (January 29, 2026): Fixed class name order mismatch between ImageFolder and ELEVATER_DATASETS
  - **Problem**: For file-based datasets, `ImageFolder` sorts class folders alphabetically (case-sensitive), putting capital-letter folders (`Faces`, `Faces_easy`, `Leopards`, `Motorbikes`) before lowercase ones (`accordion`, `airplanes`). But ELEVATER's `class_names` has its own order. This caused completely wrong label assignments (0.12% accuracy instead of 87.5% for Caltech101)
  - **Fix**: Use folder name from `dataset.classes[label]` to look up the correct ELEVATER canonical class name via a normalized mapping, instead of using `canonical_class_names[label]` directly
  - **Location**: [model_customizer.py:439-459](maveric/customization/model_customizer.py#L439-L459)
  - **Impact**: Correct baseline accuracy now restored for all file-based datasets with mixed-case folder names
- **Recommended First Steps**: Start with FER2013 (easiest to set up from Kaggle), then expand to other datasets as needed
- **Status**: ✅ **Fixed (January 29, 2026)** - File-based datasets now load correctly from manually placed test data
- **Improvement** (January 28, 2026): Added helpful error messages that display exact paths and setup instructions when test data fails to load
- **Fix** (January 29, 2026): Added ImageFolder fallback for file-based datasets ([model_customizer.py:388-398](maveric/customization/model_customizer.py#L388-L398)) - automatically loads from filesystem when torchvision handler returns None
- **Additional Fix** (January 29, 2026): Fixed FER2013 list-based class names causing `unhashable type: 'list'` error in test loader and evaluation
  - **model_customizer.py fixes**:
    - Updated `_normalize_class_name()` to handle list-based class names ([model_customizer.py:338-353](maveric/customization/model_customizer.py#L338-L353))
    - Extract canonical names before using as dictionary keys ([model_customizer.py:408-422](maveric/customization/model_customizer.py#L408-L422))
    - Fixed all dictionary operations to use canonical string names instead of lists
  - **evaluation.py fixes**:
    - Added `_get_canonical_name()` static method ([evaluation.py:27-38](maveric/customization/evaluation.py#L27-L38))
    - Fixed `_create_text_classifier_with_templates()` to extract canonical names for prompt formatting ([evaluation.py:65-67](maveric/customization/evaluation.py#L65-L67))
    - Fixed `evaluate()` method to use canonical names ([evaluation.py:137-139](maveric/customization/evaluation.py#L137-L139))
    - Fixed `evaluate_detailed()` method to use canonical names in prompts and dictionary keys ([evaluation.py:192-194, 220-222](maveric/customization/evaluation.py#L192-L194))
    - Fixed `evaluate_with_metrics()` to use canonical names for sklearn classification_report ([evaluation.py:288, 342-346](maveric/customization/evaluation.py#L288))
  - **Impact**: FER2013 (and any dataset with list-based class names) now works correctly throughout the entire pipeline

### January 7, 2026 - Domain Adaptation Implementation Complete

**Enhancement: Complete Domain Adaptation System with Visual Inspection**:
- **Purpose**: Simulate test data characteristics during training to improve model robustness
- **Core Features**:
  - **Gaussian Blur**: Simulates low quality, pixelation, or motion blur (configurable probability and sigma range)
  - **JPEG Compression**: Adds compression artifacts typical of web images (configurable quality range)
  - **Resolution Degradation**: Simulates downsampled/lower resolution images (fixed target or scale range)
  - **Applied After Augmentation**: Domain transforms applied after RandAugment for proper pipeline order
  - **Flexible Configuration**: Works with or without RandAugment, dataset-specific parameters
- **New Features** (January 7):
  - **Grid Visualization**: Save 10×10 grids of augmented/domain-adapted samples for manual inspection
  - **CLI Control**: `--save-augmented-grids` flag in `03_model_customization.py`
  - **Console Logging**: Detailed output showing augmentation and domain adaptation settings
  - **Error Handling**: Graceful fallback if domain adaptation fails
- **Bug Fixes** (January 7):
  - Fixed `_save_augmented_grids()` method placement - moved from wrong class to `ModelCustomizer` where it's called
  - Fixed FER2013 list-based class name handling in dataset classes and display code (handles `['happy', 'smiling']` format)
  - Updated `_normalize_label()` to extract canonical name from list class names
  - Fixed Country211 missing classes issue - gracefully skip countries with missing images (SM, SN, SO, SS)
- **Location**:
  - Core implementation: [model_customizer.py:1104-1155, 1157-1194](maveric/customization/model_customizer.py)
  - Grid visualization: [model_customizer.py:604-664](maveric/customization/model_customizer.py)
  - Console logging: [model_customizer.py:265-283](maveric/customization/model_customizer.py)
- **Configuration**: [maveric_config.yaml:88-107](experiments/maveric_config.yaml)
- **Benefits**:
  - **Visual Confirmation**: Inspect domain-adapted samples before training starts
  - **Transparency**: Console shows exactly what transforms are active
  - **Robustness**: Error handling prevents training crashes from bad transforms
  - **+1-2% Accuracy**: Expected improvement on degraded test sets (CIFAR-10/100, Food101)
  - **Flexible**: Easy to enable/disable, per-dataset configuration
  - **Minimal Overhead**: Only 10-15% training time increase
- **Configuration Example** (maveric_config.yaml):
  ```yaml
  training:
    use_domain_adaptation: true
    domain_blur_probability: 0.3
    domain_jpeg_probability: 0.3
    domain_downsample_probability: 0.3
    domain_target_size: 32  # CIFAR-10/100 = 32, MNIST = 28, null = use scale_range
  ```
- **CLI Usage** (with grid visualization):
  ```bash
  python experiments/03_model_customization.py \
      --input ./results/cifar10/curated/ \
      --config experiments/maveric_config.yaml \
      --save-augmented-grids
  ```
- **Console Output**:
  ```
  📦 Creating training dataset...
     Augmentation: RandAugment (num_ops=7, magnitude=22)
     Domain Adaptation: Enabled
        - Blur probability: 30.0%
        - JPEG probability: 30.0%
        - Downsample probability: 30.0%
        - Target size: 32x32 (CIFAR-10/100 mode)

  📸 Saving augmented sample grids for visual inspection...
  ✅ Saved augmented samples grid to: results/cifar10/models/augmented_grids/cifar10_augmented_grid.png
     Grid shows effects of:
     - RandAugment (ops=7, mag=22)
     - Domain Adaptation (blur/JPEG/downsample)
  ```
- **Grid Output Location**: `{checkpoint_dir}/../augmented_grids/{dataset_name}_augmented_grid.png`
- **Testing**: [test_domain_adaptation.py](test_domain_adaptation.py) - Comprehensive test suite
- **Documentation**:
  - [DOMAIN_ADAPTATION_IMPLEMENTATION.md](DOMAIN_ADAPTATION_IMPLEMENTATION.md) - Initial implementation
  - [DOMAIN_ADAPTATION_IMPROVEMENTS_SUMMARY.md](DOMAIN_ADAPTATION_IMPROVEMENTS_SUMMARY.md) - Complete feature list

### January 4, 2026 - Keep Count Feature for Mahalanobis Filter

**Enhancement: Dual Input Method for Sample Selection**:
- **Purpose**: Allow users to specify filtering criteria using either percentages or exact counts
- **New Features**:
  - **Keep Count Input**: IntText widget for entering exact sample count (e.g., 350 for exactly 350 samples)
  - **Bidirectional Sync**: Percentile ↔ Count auto-update in real-time
  - **Priority Logic**: Keep Count takes priority when specified (> 0), falls back to Keep Percentile otherwise
  - **Context-Aware**: Automatically updates when switching modes or classes
  - **Exact Filtering**: User gets **exactly** the number of samples requested (not approximate)
- **Location**: [interactive.py:1366-1491, 1493-1582, 1764-1906, 2033-2162](maveric/visualization/interactive.py)
- **Modified Methods**:
  - `_apply_mahalanobis_filter()`: Now accepts `keep_count` parameter, uses it when specified
  - `_apply_mahalanobis_filter_class_based()`: Now accepts `keep_count` parameter, uses it when specified
  - Apply Filter callback: Passes both percentile and count to filtering methods
- **Benefits**:
  - **Flexible Input**: Think in percentages (30%) or absolute counts (350 samples)
  - **Precision**: Get exactly N samples (no rounding errors)
  - **User-Friendly**: See both values, change either one
  - **Automatic**: No mental math required
- **Usage Example**:
  ```python
  # Global mode: Enter 5000 in Keep Count → filters to exactly 5,000 samples
  # Keep Percentile auto-updates to 10.0 (if dataset has 50,000 samples)

  # Class-Based mode: Enter 350 in Keep Count → filters class to exactly 350 samples
  # Keep Percentile auto-updates to corresponding percentage for that class
  ```
- **Console Output**:
  ```
  🎯 Keeping exactly 350 samples (requested: 350)
  ✅ Kept 350 / 5,000 samples for class 'airplane'
  ```
- **Documentation**: [KEEP_COUNT_IMPLEMENTATION.md](KEEP_COUNT_IMPLEMENTATION.md)

### December 22, 2025 - Mahalanobis Filter Global & Class-Based Modes

**Enhancement: Dual-Mode Mahalanobis Filtering System**:
- **Purpose**: Support both global (all classes) and class-based (per-class) filtering workflows
- **Two Modes**:
  1. **Global Mode**: Filter all data at once with uniform settings (existing functionality)
  2. **Class-Based Mode**: Filter each class individually with custom settings (NEW)
- **Class-Based Mode Features**:
  - **Class Selector**: Dropdown to choose which class to filter
  - **Percentile Controls**: Configure weighted %ile, consistency %ile, and keep %ile per class
  - **Apply Filter**: Generate class-specific analysis plot and statistics
  - **Add Data**: Accumulate filtered data from multiple classes
  - **Save Filtered Data**: Export grid PNG files (format: `datasetName_className_###.png`)
  - **Reset Button**: Clear filtered data and return to original state (NEW)
- **Location**: [interactive.py:1299-1567](maveric/visualization/interactive.py#L1299-L1567)
- **New Methods**:
  - `_apply_mahalanobis_filter_class_based()`: [interactive.py:1832-1945](maveric/visualization/interactive.py#L1832-L1945)
  - `_plot_mahalanobis_analysis_class_based()`: [interactive.py:1947-2060](maveric/visualization/interactive.py#L1947-L2060)
  - `_save_class_filtered_grids()`: [interactive.py:2062-2154](maveric/visualization/interactive.py#L2062-L2154)
  - `_consolidate_class_based_data()`: [interactive.py:2156-2192](maveric/visualization/interactive.py#L2156-L2192)
- **Benefits**:
  - **Flexible**: Choose global or per-class filtering based on workflow needs
  - **Targeted**: Apply different quality criteria to each class
  - **Visual**: Save grid PNGs for manual inspection per class
  - **Iterative**: Review and adjust each class before adding to training set
  - **Compatible**: Consolidated data works with Balance tab and other features
- **Usage - Global Mode**:
  ```python
  from maveric.visualization import start_interactive_gui
  gui = start_interactive_gui('cifar10')
  # Navigate to Tab 2: Mahalanobis Filter
  # Select "Global" mode
  # Set Weighted %ile (e.g., 95)
  # Set Consistency %ile (e.g., 95)
  # Set Keep %ile (e.g., 30)
  # Click "Apply Filter"
  # → All classes filtered at once
  # → Data immediately available for Balance tab
  ```
- **Usage - Class-Based Mode**:
  ```python
  from maveric.visualization import start_interactive_gui
  gui = start_interactive_gui('cifar10')
  # Navigate to Tab 2: Mahalanobis Filter
  # Select "Class-Based" mode
  # For each class:
  #   1. Select class from dropdown (e.g., "airplane")
  #   2. Set Weighted %ile (e.g., 95)
  #   3. Set Consistency %ile (e.g., 95)
  #   4. Set Keep %ile (e.g., 40)
  #   5. Click "Apply Filter" → See class-specific plot
  #   6. Click "Add Data" → Store filtered class
  #   7. (Optional) Click "Save Filtered Data" → Export grids
  #   8. (Optional) Click "Reset" → Clear specific class or all data
  # After all classes added:
  # → Consolidated data available for Balance tab
  ```
- **Reset Button Behavior**:
  - **Global Mode**: Restores data before Mahalanobis filter (undo filter)
  - **Class-Based (no class)**: Clears ALL accumulated class data (start over)
  - **Class-Based (specific class)**: Removes that class, re-consolidates remaining data
- **Documentation**:
  - [MAHALANOBIS_CLASS_BASED_MODE.md](MAHALANOBIS_CLASS_BASED_MODE.md)
  - [MAHALANOBIS_RESET_BUTTON.md](MAHALANOBIS_RESET_BUTTON.md)

### December 21, 2025 - Mahalanobis Filter Tab Simplification

**Enhancement: Simplified and Enhanced Mahalanobis Filter Tab**:
- **Purpose**: Cleaner UI with configurable ideal point selection
- **Changes**:
  1. **Per-class mode removed**: Global mode only (hardcoded `per_class=False`)
  2. **Explanation section removed**: No verbose text at top of tab
  3. **Percentile controls added**: Two text boxes for ideal point configuration
     - **Weighted %ile**: Percentile for weighted_class_score (default: 95)
     - **Consistency %ile**: Percentile for consistency (default: 95)
  4. **Percentage dropdown removed**: Only "Keep %ile" text box remains
  5. **Histogram scaling fixed**: Added `density=True` for proper normalization
- **New Controls**:
  - **Weighted %ile** (1-99%): Configure ideal point for weighted_class_score axis
  - **Consistency %ile** (1-99%): Configure ideal point for consistency axis
  - **Keep %ile** (1-99%): Percentage of samples to keep (closest to ideal point)
- **Location**: [interactive.py:1299-1424](maveric/visualization/interactive.py#L1299-L1424)
- **Filter Logic**: [interactive.py:1426-1580](maveric/visualization/interactive.py#L1426-L1580)
- **Benefits**:
  - **Cleaner UI**: Removed explanation and mode selector
  - **Flexible ideal point**: User can adjust target percentiles for both axes
  - **Simpler workflow**: Only global mode, fewer decisions
  - **Better visualizations**: Histogram density normalization shows true distributions
- **Usage**:
  ```python
  from maveric.visualization import start_interactive_gui
  gui = start_interactive_gui('cifar10')
  # Navigate to Tab 2: Mahalanobis Filter
  # Set Weighted %ile (e.g., 95)
  # Set Consistency %ile (e.g., 95)
  # Set Keep %ile (e.g., 30)
  # Click "Apply Filter"
  # View normalized histograms and ellipse plot
  ```

### December 21, 2025 - Balance Settings Tab Improvements

**Enhancement: Simplified and Enhanced Balance Settings Tab**:
- **Purpose**: Streamlined UI and flexible sample sorting for balanced datasets
- **Changes**:
  1. **Min Samples removed**: Removed from UI, now hardcoded to 1 (no class filtering)
  2. **Oversampling checkbox visibility**: Set explicit width (500px) for full visibility
  3. **Sorting method selector**: New dropdown to choose sample sorting strategy
- **New Sorting Options**:
  - **Consistency** (default): Sort samples by consistency score (higher = better)
  - **Weighted**: Sort samples by weighted_class_score (higher = better)
  - **Impact**: Choose which metric to prioritize when selecting best samples during balancing
- **Location**: [interactive.py:1973-2011](maveric/visualization/interactive.py#L1973-L2011)
- **Balancing Logic**: [interactive.py:439-542](maveric/visualization/interactive.py#L439-L542)
- **Benefits**:
  - **Cleaner UI**: One less widget, simpler interface
  - **Flexible selection**: Choose consistency or weighted score for sample ranking
  - **Better visibility**: Oversampling checkbox fully visible with explicit layout
  - **No class removal**: min_samples=1 ensures all classes are kept
- **Usage**:
  ```python
  from maveric.visualization import start_interactive_gui
  gui = start_interactive_gui('cifar10')
  # Navigate to Tab 4: Balance Settings
  # Select strategy (e.g., "median")
  # Select sorting method: "Consistency" or "Weighted"
  # Check "Enable Oversampling" if needed
  # Click "Apply Balance"
  ```

### December 19, 2025 - Mahalanobis Distance Filtering

**New Feature: Mahalanobis Distance Filtering Tab in Interactive GUI**:
- **Purpose**: Advanced sample selection using Mahalanobis distance from ideal point
- **Implementation**: New "Mahalanobis Filter" tab between "Quality Thresholds" and "EfficientNet Prediction"
  - **Location**: [interactive.py:1293-1679](maveric/visualization/interactive.py#L1293-L1679)
  - **Tab order**: Metric Weights → Quality Thresholds → **Mahalanobis Filter** → EfficientNet → Balance Settings
- **Algorithm**:
  - Jointly optimizes `weighted_class_score` and `consistency` metrics
  - Calculates ideal point at 95th percentile of both metrics
  - Computes Mahalanobis distance accounting for correlation and different scales
  - Keeps top N% of samples closest to ideal point
- **Features**:
  - **Dropdown selector**: Choose percentage (10%, 20%, 30%, 40%, 50%)
  - **Custom input**: Enter any percentage (1-99%) for precise control
  - **Filter modes**: Global (all samples) or Per-Class (balanced per class)
  - **Visual feedback**: XY scatter plot with ellipse boundary, marginal histograms, correlation coefficient
  - **Statistics**: Before/after sample counts, class distribution with total class count
- **Visualization**:
  - Green dots: Selected samples
  - Gray dots: Rejected samples
  - Red star: Ideal point (95th percentile)
  - Red ellipse: Selection boundary (Mahalanobis distance threshold)
  - ρ value: Correlation coefficient displayed in corner
  - Marginal plots: Distribution of each metric (all vs selected)
- **Benefits**:
  - **Better selection**: Jointly optimizes both metrics (vs independent thresholds)
  - **More samples**: Typically retains 20-40% more samples than simple thresholds
  - **Higher quality**: Selected samples have better mean/min scores on both metrics
  - **Flexible**: Easy percentage adjustment via dropdown or text input
  - **Automatic reset**: Prevents compounding filters by restoring pre-Mahalanobis data on each apply
- **Automatic Reset Behavior**:
  - First filter: Backs up data before applying Mahalanobis filter
  - Change percentage: Automatically resets to backup before applying new filter
  - **Prevents compounding**: Always filters from same baseline (data after Tab 1)
  - **Example**: Apply 30% (50K→15K), change to 20% → resets to 50K, then filters to 10K (NOT 15K→3K)
- **Usage**:
  ```python
  from maveric.visualization import start_interactive_gui
  gui = start_interactive_gui('cifar10')
  # 1. Go to Tab 1: Quality Thresholds, click "Apply Settings"
  # 2. Navigate to Tab 2: Mahalanobis Filter
  # 3. Select percentage (e.g., 30%) and mode (Global/Per-Class)
  # 4. Click "Apply Filter"
  # 5. View XY plot with ellipse and statistics
  # 6. Change percentage and re-apply → automatically resets first
  ```

### December 18, 2025 - CIFAR-100 Class Name Fix & Intelligent Balancing

**Bug Fix 1: CIFAR-100 Class Ordering Mismatch (CRITICAL - Baseline Accuracy 1.15% → 65.1%)**:
- **Issue**: Baseline model evaluation showed 1.15% accuracy instead of expected 65.1% for CIFAR-100
  - **Root cause**: ELEVATER class_names list was in custom order (`['chimpanzee', 'trout', 'skunk', ...]`) but torchvision's CIFAR-100 uses alphabetically sorted order (`['apple', 'aquarium_fish', 'baby', ...]`)
  - **Impact**: Test sample labels were completely mismatched (e.g., torchvision label 0 = 'apple' but code interpreted as 'chimpanzee')
  - **Symptom**: Baseline accuracy dropped from ~65% to ~1% (random chance for 100 classes)
- **Fix**: Updated CIFAR-100 class_names to match torchvision's exact alphabetical ordering
  - **Location**: [elevater_datasets.py:153-175](maveric/datasets/elevater_datasets.py#L153-L175)
  - **Ordering**: Alphabetically sorted with underscores converted to spaces for REACT style
  - **First 10 classes**: `['apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle']`
  - **Result**: Class labels now correctly aligned with torchvision dataset
- **Key insight**: Torchvision CIFAR-100 uses `.classes` attribute with alphabetically sorted class names

**Bug Fix 2: CIFAR-100 Missing 9 Classes with Spaces (Interactive GUI)**:
- **Issue**: Interactive GUI only detected 91 out of 100 CIFAR-100 classes
  - **Root cause**: Predefined class names used underscores (`'aquarium_fish'`, `'sweet_pepper'`) but JSON columns have spaces (`"Class_aquarium fish_img2img"`, `"Class_sweet pepper_txt2txt"`)
  - **Impact**: 9 classes with spaces were not recognized: `'aquarium fish'`, `'lawn mower'`, `'maple tree'`, `'oak tree'`, `'palm tree'`, `'pickup truck'`, `'pine tree'`, `'sweet pepper'`, `'willow tree'`
- **Fix**: Updated `cifar100_class_names` to match ELEVATER_DATASETS exactly (with spaces)
  - **Location**: [interactive.py:55-69](maveric/visualization/interactive.py#L55-L69)
  - **Result**: All 100 CIFAR-100 classes now correctly detected
- **Key insight**: Column names in JSON files preserve spaces from ELEVATER class names

**Enhancement: Intelligent Sample Selection in Balance CLI**:
- **Issue**: `balance_cli.py` used random sampling, making results unpredictable and ineffective
  - **Impact**: Could keep low-quality samples while discarding high-quality ones
- **Fix**: Implemented consistency-based sample selection matching `interactive.apply_balance()`
  - **Location**: [balance_cli.py:63-174](maveric/utils/balance_cli.py#L63-L174)
  - **Algorithm**:
    1. Sort samples by `consistency` score (higher = better quality)
    2. Undersampling: Keep top N samples with highest consistency
    3. Oversampling: Duplicate best samples cyclically
    4. Shuffle final dataset with fixed seed (42) for reproducibility
- **Benefits**:
  - Quality-preserving: Always keeps best samples based on consistency scores
  - Predictable results: Not random - highest quality samples always selected
  - Effective balancing: Same proven algorithm as interactive GUI

### December 13, 2025 - Visual Grid Export & Manual Balancing CLI

**New Feature 1: Automatic Grid Visualization on Save**:
- **Purpose**: Enable visual inspection of curated data without loading individual images
- **Implementation**: Added `save_sample_grids()` method to interactive GUI
  - **Location**: [interactive.py:852-995](maveric/visualization/interactive.py#L852-L995)
  - **Integration**: [interactive.py:1706-1714](maveric/visualization/interactive.py#L1706-L1714)
- **Note**: As of July 6, 2026, this feature is **disabled by default** for performance. Enable with `save_grid_visualization: true` in config.
- **Functionality**:
  - Generates 10×10 image grids when "Save Data" button is clicked (if enabled)
  - **Organized by class**: Images sorted by label for easy class-by-class inspection
  - Each grid contains 100 images with labels and quality scores
  - Saves to `curationResults/` folder alongside training JSON files
  - File naming: `{dataset_name}_grid_{number:03d}.png`
- **Features**:
  - **OPTIMIZED**: Loads from dataset-specific `images/` folder (local, fast)
  - **NOT** from global cache (avoids Google Drive/NFS latency)
  - Uses same images copied by `_copy_training_images()`
  - Displays compact info: ID, label, weighted score, consistency
  - Creates multiple grids for >100 samples (e.g., 53 grids for 5,250 samples)
  - 30×30 inch figure size with 150 DPI for high-quality output
- **Usage**:
  ```python
  # Automatic: Click "Save Data" button in GUI
  from maveric.visualization import start_interactive_gui
  gui = start_interactive_gui('cifar10')
  # → Saves both JSON and PNG grids

  # Manual: Programmatic call
  gui.apply_thresholds()
  grid_path = gui.save_sample_grids()
  # → Returns path to curationResults folder
  ```
- **Benefits**:
  - **Class-organized**: All images of same class grouped together for easy inspection
  - Quick quality check without loading all images individually
  - Easy sharing/documentation of curation results
  - Visual confirmation of data quality before training
  - Compact format: 100 images per PNG file
  - **Fast on network drives**: Local I/O only, no global cache access

**New Feature 2: CLI for Manual Dataset Balancing**:
- **Purpose**: Balance manually cleaned training datasets after visual inspection
- **Implementation**: New CLI tool for post-curation dataset balancing
  - **Location**: [balance_cli.py](maveric/utils/balance_cli.py) (main implementation)
  - **Standalone script**: [balance_dataset.py](balance_dataset.py) (wrapper for easy execution)
  - **Test suite**: [test_balance_cli.py](test_balance_cli.py) (comprehensive testing)
- **Workflow Integration**:
  1. Curate data using interactive GUI (Save Data)
  2. Manually inspect grids in `curationResults/`
  3. Remove bad samples from JSON files
  4. **Balance cleaned data using this CLI tool** ← NEW
  5. Use balanced data for model customization
- **Balancing Strategies**:
  - `min`: Balance to smallest class (pure undersampling)
  - `max`: Balance to largest class (pure oversampling)
  - `mean`: Balance to average class size (mixed sampling)
  - `median`: Balance to median class size (mixed sampling)
  - `custom`: Balance to user-specified target per class
- **Intelligent Sample Selection** (matches `interactive.apply_balance()` behavior):
  - **Sorts by consistency score**: Samples ranked by quality (higher = better)
  - **Undersampling**: Keeps TOP N samples with highest consistency (not random)
  - **Oversampling**: Duplicates best samples cyclically (not random)
  - **Shuffling**: Final dataset shuffled with fixed seed for reproducibility
  - **Result**: Balanced datasets maintain highest quality samples
- **Features**:
  - Accepts single JSON file or directory with multiple rotation files
  - Independent control over undersampling and oversampling
  - Rotation file support (splits large datasets into chunks)
  - Dry run mode for preview without saving
  - Detailed progress reporting per class
  - Warning if 'consistency' column missing
- **Usage Examples**:
  ```bash
  # Balance using minimum class size (pure undersampling)
  python balance_dataset.py \
      --input ./curated_data \
      --output ./balanced_data \
      --strategy min

  # Balance to 500 samples per class (custom target)
  python balance_dataset.py \
      --input ./curated_data \
      --strategy custom \
      --target-per-class 500 \
      --enable-oversampling \
      --output ./balanced_data

  # Balance using mean with rotation files
  python balance_dataset.py \
      --input ./curated_data \
      --strategy mean \
      --enable-oversampling \
      --rotation-size 1000 \
      --output ./balanced_data

  # Dry run to preview changes
  python balance_dataset.py \
      --input ./curated_data \
      --strategy median \
      --dry-run \
      --output ./balanced_data

  # Alternative: Run via module
  python -m maveric.utils.balance_cli --input ./data --strategy min --output ./balanced
  ```
- **Benefits**:
  - **Quality-preserving**: Always keeps best samples based on consistency scores
  - **Predictable results**: Not random - highest quality samples always selected
  - **Effective balancing**: Same proven algorithm as interactive GUI
  - Complete control over dataset balancing after manual cleanup
  - Flexible strategies matching interactive GUI capabilities
  - Handles imbalanced datasets from manual inspection

### December 12, 2025 - Training Evaluation Consistency Fix

**Enhancement: REACT-Style Template Ensembling in Training Loop**:
- **Issue**: Training loop used single-template evaluation while final evaluation used REACT-style template ensembling
  - **Impact**: Accuracy reported during training (e.g., 91.82%) differed from final evaluation (92.06%)
  - **Example gap**: 0.24% difference (24 samples out of 10,000 for CIFAR-10)
  - **Root cause**: Training used `"a photo of a {}"` while final used 18 templates with ensembling
- **Fix**: Training loop now uses same REACT-style template ensembling as final evaluation
  - **Location**: [training.py:39-87](maveric/customization/training.py#L39-L87)
  - **Integration**: [model_customizer.py:149-157](maveric/customization/model_customizer.py#L149-L157)
  - **Benefits**:
    - Consistent accuracy numbers between epochs and final evaluation
    - More accurate progress monitoring during training
    - Same evaluation method (REACT-style) used throughout
  - **Tradeoff**: Slightly slower per-epoch evaluation (~18x for CIFAR-10 with 18 templates)
- **Backward compatibility**: Falls back to single-template if templates not provided

### December 7, 2025 - Critical Bug Fixes

**Bug Fix: Class Names with Special Characters in File Paths**:
- **Issue**: GTSRB class name `"end / de-restriction of 80 kph speed limit"` contains `/` which Linux interprets as directory separator
- **Impact**: Reference image caching failed with `FileNotFoundError` for datasets with special characters in class names
- **Fix**: Added `sanitize_filename()` function to replace problematic characters (`/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`) with underscores
- **Location**: [cache_manager.py:17-74](maveric/retrieval/cache_manager.py#L17-L74)
- **Applied**: Reference image caching and file-based dataset loading
- **Affected datasets**: GTSRB, any future datasets with filesystem-reserved characters

**Bug Fix: FER2013 List-Based Class Names (Multiple Fixes)**:
- **Issue 1**: FER2013 uses lists of synonyms for class names (e.g., `['happy', 'smiling']`) instead of single strings
  - **Impact**: `sanitize_filename()` received a list and failed with `AttributeError: 'list' object has no attribute 'replace'`
  - **Fix**: Updated `sanitize_filename()` to handle both strings and lists (uses first element as canonical name)
  - **Example**: `['happy', 'smiling']` → `'happy'`, `['sad', 'depressed']` → `'sad'`
  - **Location**: [cache_manager.py:46-51](maveric/retrieval/cache_manager.py#L46-L51)

- **Issue 2**: Dictionary keys cannot be lists (unhashable type)
  - **Impact**: Code attempted to use list class names as dictionary keys: `reference_samples[class_name]` where `class_name = ['happy', 'smiling']`
  - **Error**: `TypeError: unhashable type: 'list'`
  - **Fix**: Extract canonical name (first element) before using as dictionary key at all assignment locations
  - **Pattern**: `canonical_name = class_name[0] if isinstance(class_name, list) else class_name`
  - **Locations in elevater_datasets.py**:
    - [Line 1581-1582](maveric/datasets/elevater_datasets.py#L1581-L1582) - Torchvision reference samples initialization
    - [Line 1617](maveric/datasets/elevater_datasets.py#L1617) - Torchvision reference samples assignment
    - [Line 1651-1652](maveric/datasets/elevater_datasets.py#L1651-L1652) - File-based reference samples initialization
    - [Line 1687](maveric/datasets/elevater_datasets.py#L1687) - File-based reference samples assignment
  - **Locations with synonym expansion**:
    - **retriever.py** [Line 242-258](maveric/retrieval/retriever.py#L242-L258) - Text embedding creation with synonym expansion
    - **cache_manager.py** [Line 500-512](maveric/retrieval/cache_manager.py#L500-L512) - Reference text caching with synonym expansion
    - **Special handling**: For FER2013 lists like `['happy', 'smiling']`, creates prompts for ALL synonyms to get richer embeddings
    - **Example**: `['happy', 'smiling']` with template `"a photo of a {}"` generates prompts: `["a photo of a happy", "a photo of a smiling"]`

- **Affected datasets**: FER2013 (only dataset with list-based class names)

### November 20, 2025 - Critical Performance & Evaluation Fixes

**Performance Fix 1: sklearn Import Bottleneck (10-20x Speedup)**:
- **Issue**: `sklearn.metrics.pairwise.cosine_similarity` imported inside retrieval loop
  - **Impact**: 10-20 seconds per sample overhead (importing on every class iteration)
  - **For CIFAR-10**: 10 classes × 1-2s import = 10-20s per sample
  - **Root cause**: Import statement inside `compute_sample_scores()` method
- **Fix**: Moved import to module level
  - **Location**: [retriever.py:11](maveric/retrieval/retriever.py#L11) (module level)
  - **Removed from**: Line 556 (inside loop - deleted)
  - **Impact**: Eliminated 10-20s overhead per sample
- **Performance improvement**: 10-20x faster data retrieval

**Performance Fix 2: Google Drive Cache Bottleneck (Critical)**:
- **Issue**: Google Drive FUSE/NFS makes `.exists()` file checks take 10+ seconds each
  - **Symptom**: Every sample took 10-20 seconds regardless of cache hit/miss or CPU/GPU usage
  - **Debug output**: Cache check taking 10.8s-12.5s (99.7% of total time)
  - **Impact**: 2-3 file checks per sample = 20-30+ seconds overhead
- **Root cause**: Google Drive network filesystem latency for file existence checks
- **Solutions implemented**:
  1. **Configuration fix**: Added cache disable options
     - `enable_sample_cache: false` to skip sample metadata cache
     - `enable_image_cache: false` to skip image cache
  2. **Redundant check optimization**: Skip image cache check after sample cache miss
     - **User discovery**: "Why do we need another .exists() check at Cache MISS?"
     - Added `skip_cache_check=True` parameter to `download_and_cache_image()`
     - **Location**: [cache_manager.py:225, 240-243](maveric/retrieval/cache_manager.py#L225)
     - **Integration**: [retriever.py:433](maveric/retrieval/retriever.py#L433)
- **Recommendation**: Use local disk for cache instead of Google Drive when possible
- **Performance improvement**: 10-20x faster when caches disabled on Google Drive

**Configuration Bug Fixes (Multiple Critical Issues)**:
- **Issue**: Configuration values not propagated from YAML to MAVERIC initialization
  - **Symptoms**:
    - Timeouts not working (15s downloads despite `request_timeout: 1`)
    - Retries not applied (stuck on failed downloads)
    - Cache settings ignored
- **Fixes in 01_data_retrieval.py**:
  1. **max_retries**: Now properly passed from config
     - **Location**: [01_data_retrieval.py:265](experiments/01_data_retrieval.py#L265)
     - Previous: Used default value (3) always
  2. **request_timeout**: Now properly passed from config
     - **Location**: [01_data_retrieval.py:266](experiments/01_data_retrieval.py#L266)
     - Previous: Used default value (5) always
  3. **enable_sample_cache**: Now properly passed from config
     - **Location**: [01_data_retrieval.py:262](experiments/01_data_retrieval.py#L262)
     - Previous: Missing parameter (always enabled)
  4. **enable_image_cache**: Fixed path to read from top-level config
     - **Location**: [01_data_retrieval.py:261](experiments/01_data_retrieval.py#L261)
     - Previous: Read from wrong nested path
  5. **Added logging**: Cache and timeout settings now logged at startup
     - **Location**: [01_data_retrieval.py:276-277](experiments/01_data_retrieval.py#L276-L277)
- **Testing**: Confirmed all config values now applied correctly

**REACT-Style Evaluation Implementation**:
- **Issue**: Oxford Pets showing 75.99% accuracy vs REACT's 87.1%
  - **Root causes**: Missing template ensembling, wrong class name format, tensor shape bugs
- **Fix 1: Template Ensembling with Double Normalization**
  - Implemented REACT-style template ensembling in evaluation
  - **Method**: `_create_text_classifier_with_templates()`
  - **Location**: [evaluation.py:27-75](maveric/customization/evaluation.py#L27-L75)
  - **Process**:
    1. Generate prompts for each class using all templates
    2. Compute embeddings for each prompt
    3. Normalize each embedding (L2 norm)
    4. Average embeddings for the class
    5. Re-normalize the averaged embedding
  - **Integration**: Updated `evaluate()` and `evaluate_detailed()` to accept templates
  - **Tensor shape fix**: Changed `torch.stack(dim=1)` to `dim=0` (line 73)
    - Previous: Created (embedding_dim, num_classes) → mat mul error
    - Fixed: Creates (num_classes, embedding_dim) → correct for `image_embeds @ text_features.T`

- **Fix 2: Class Name Normalization System**
  - **Problem**: Training data had lowercase/underscores, test data had Title Case/spaces
  - **Impact**: "Excluding 27 classes not in training data" error
  - **Solution**: Normalized matching with proper ELEVATER class names for embeddings
  - **Implementation**:
    - `_normalize_class_name()` helper for flexible matching
    - **Location**: [model_customizer.py:287-297](maveric/customization/model_customizer.py#L287-L297)
    - Normalizes to lowercase with underscores for comparison only
    - **Key insight**: Use ELEVATER_DATASETS class names for CLIP embeddings (not normalized)
  - **Template retrieval**: `_get_dataset_templates()` method
    - **Location**: [model_customizer.py:467-490](maveric/customization/model_customizer.py#L467-L490)
    - Loads dataset and extracts templates automatically

- **Fix 3: Oxford Pets Class Names Updated**
  - **Problem**: Torchvision uses Title Case for all classes, not mixed case
  - **Impact**: Class name mismatch between REACT format and torchvision
  - **Solution**: Updated class names to match torchvision exactly
  - **Location**: [elevater_datasets.py:239-247](maveric/datasets/elevater_datasets.py#L239-L247)
  - **Example**: 'Abyssinian', 'American Bulldog', 'British Shorthair' (all Title Case)
  - **Templates updated**: Added periods to Oxford Pets templates (line 736-737)

- **Expected improvement**: 75.99% → ~87.1% (matching REACT benchmark)

**Debug Tools Created**:
- **debug_retrieval_timing.py**: Comprehensive timing instrumentation
  - Patches `compute_sample_scores()` with detailed step-by-step timing
  - Shows percentage breakdown of time spent in each operation
  - Identifies bottlenecks with warnings (>50% time in single step)
- **test_cache_performance.py**: Standalone cache I/O performance testing
  - Tests JSON read/write/decode performance
  - Bypasses MAVERIC to isolate cache issues
  - Provides baseline performance measurements
- **POTENTIAL_BOTTLENECKS_CHECKLIST.md**: Investigation guide
  - Lists all potential performance issues
  - Prioritized by likelihood
  - Includes solutions for each issue
- **DEBUG_TIMING_INSTRUCTIONS.md**: Usage instructions for debug tools

### November 24, 2025 - Documentation & Visualization

**New Documentation**:
- **Pipeline Visualization**: Created comprehensive SVG diagram of MAVERIC architecture
  - **Location**: [docs/maveric_pipeline.svg](docs/maveric_pipeline.svg)
  - **Content**: Complete 4-stage pipeline with reference generation subsystem
  - **Style**: White background, clear color coding for components/data/cache flows
- **Reference Generation Guide**: Complete documentation of reference system
  - **Location**: [docs/REFERENCE_GENERATION.md](docs/REFERENCE_GENERATION.md)
  - **Coverage**: Sample selection, REACT templates, CLIP embeddings, caching, usage
  - **Details**: 4-step process with examples, troubleshooting, and best practices

### November 23, 2025 - Architecture Cleanup

**Deprecated Code Removal**:
- **Removed**: `maveric/interactive/` folder entirely (redundant components)
  - Deleted: `threshold_selector.py`, `quality_dashboard.py`, `widgets.py`, `__init__.py`
  - **Reason**: `maveric.visualization.interactive.py` provides superior full-featured GUI
  - **Impact**: Simplified codebase, single source of truth for interactive features
- **Removed**: `MAVERIC.launch_dashboard()` method
  - **Replacement**: Use `from maveric.visualization import start_interactive_gui` directly
  - No backward compatibility layer - clean break for better maintainability
- **Removed**: `create_interactive_gui()` legacy wrapper in `visualization/__init__.py`
  - **Replacement**: Use `start_interactive_gui()` directly
- **Removed**: Legacy `quality_thresholds` field mapping in `config.py`
  - **Current**: Use `default_thresholds` parameter directly
- **Fixed**: Incorrect "Legacy" comment for `weighted_class_score` in `config.py`
  - **Clarification**: `weighted_class_score` is an active metric, not deprecated

**Migration Guide**:
```python
# OLD (no longer works):
dashboard = maveric.launch_dashboard(retrieval_result)
# or
from maveric.visualization import create_interactive_gui
gui = create_interactive_gui('cifar10')

# NEW (correct):
from maveric.visualization import start_interactive_gui
gui = start_interactive_gui('cifar10', config_file=None)
```

### November 21, 2025 - Critical Evaluation Fixes

**Fix 1: Class Name Capitalization Bug (COMPREHENSIVE FIX)**
- **Critical bug**: Class names were lowercase in evaluation, causing 4-5% accuracy loss
  - **Impact**: Oxford Pets evaluation improved from 82-83% → 87%+
  - **Root cause**: Training JSON had lowercase labels, AND torchvision's OxfordIIITPet dynamically generates ALL Title Case class names (e.g., "American Bulldog") which differs from REACT's mixed-case format
  - **Example**: `"a photo of a abyssinian"` (wrong) vs `"a photo of a Abyssinian"` (correct per REACT)
  - **Solution**: Four-part fix to ensure exact REACT class names throughout the pipeline:
    1. **Load class names directly from ELEVATER_DATASETS** ([03_model_customization.py:331-363](experiments/03_model_customization.py#L331-L363))
       - Load from `ELEVATER_DATASETS` dictionary, NOT from dataset handler
       - Avoids torchvision overriding with its own dynamically-generated class names
       - Uses EXACT REACT class names with proper mixed-case format (e.g., 'Abyssinian', 'american bulldog')
    2. **Pass class names to customize_model()** ([03_model_customization.py:415](experiments/03_model_customization.py#L415), [main.py:266-341](maveric/main.py#L266-L341))
       - Added `class_names` parameter to `customize_model()` method
       - Ensures ELEVATER class names flow through to evaluation (not training data labels)
       - Previously extracted class names from training data (normalized/lowercase)
    3. **Use class names in test loader** ([model_customizer.py:325](maveric/customization/model_customizer.py#L325))
       - Changed `_create_test_loader` to use passed `class_names` parameter
       - Previously used `test_dataset_handler.class_names` (torchvision's dynamic names)
       - Now uses EXACT REACT class names for test sample creation and evaluation
    4. **Case-insensitive label mapping in training** ([model_customizer.py:847-850](maveric/customization/model_customizer.py#L847-L850))
       - Create normalized mapping: `{'abyssinian': 0}` and `{'Abyssinian': 0}` both work
       - Handles training JSON having lowercase while evaluation uses REACT's mixed-case format
       - Normalized label lookup during training ([model_customizer.py:1025-1030](maveric/customization/model_customizer.py#L1025-L1030))
  - **Key insight**: CLIP trained on proper English grammar; REACT uses specific mixed-case format that must be matched exactly
  - **Testing**: Standalone code verified 87.19% (proper case) vs 82.28% (lowercase) on same data
  - **Consistency**: Now uses exact ELEVATER/REACT dataset class names for all operations

**Fix 2: CLIP Image Preprocessing**
- **Critical fix**: Fixed image preprocessing to use default CLIP processor behavior
  - **Impact**: Fixed aspect ratio distortion causing ~6% accuracy gap
  - **Root cause**: Explicitly setting `size={"height": 224, "width": 224}` distorted aspect ratios before center cropping
  - **Correct preprocessing**: Resize shortest edge to 224 (preserving aspect ratio), then center crop to 224x224
  - **Solution**: Use processor's default parameters instead of explicit size/crop parameters
  - **Location**: `_safe_process_images()` in [model_customizer.py:670-688](maveric/customization/model_customizer.py#L670-L688)
  - **Consistency**: Now matches standard CLIP preprocessing and published benchmarks

### November 20, 2025 - REACT-Style Text Prompting & Training Optimizations
- **Dataset-specific text templates**: Implemented REACT benchmark-style class-specific prompting
  - **Custom templates for 15+ datasets**: DTD, EuroSAT, FER2013, Food101, GTSRB, Oxford Flowers102, Oxford Pets, CIFAR-10/100, and more
  - **Multiple templates per dataset**: Provides prompt diversity for better retrieval
  - **Placeholder-based formatting**: Uses `{}` for class name insertion (e.g., "a photo of a {} texture")
  - **Location**: `get_text_templates()` method in [elevater_datasets.py](maveric/datasets/elevater_datasets.py)
  - **Integration**: Automatic template retrieval in evaluation and model customization
- **Class name normalization**: Added intelligent matching between dataset class names and template placeholders
  - **Handles format variations**: Lowercasing, underscore/hyphen to space conversion
  - **Ensures proper matching**: Training data classes align with evaluation templates
  - **Location**: [model_customizer.py](maveric/customization/model_customizer.py)
- **Training hyperparameter updates** in `experiments/maveric_config.yaml`:
  - Epochs: 10 → 20
  - Learning rate: 0.0000006 → 0.0000007
  - Weight decay: 0.05 → 0.07
  - Regularization weight: 0.80 → 0.75
  - Augmentation strength: 4 → 7
  - Augmentation magnitude: 15 → 22
  - Gradient clip value: 0.5 → 0.75
  - **Impact**: Improved model performance based on empirical experiments

### November 5, 2025 - CLIP Embedding Caching
- **Enhanced sample caching**: CLIP embeddings now cached alongside metrics
  - **Cache version upgraded to v3**: Includes CLIP image/text embeddings (base64 encoded)
  - **Performance Impact**: 80-95% speedup for subsequent dataset retrievals (vs 60-85% without embeddings)
  - **Storage**: ~17KB per sample (~500 bytes metrics + ~16KB embeddings)
  - **Total for 270K samples**: ~4.5GB (increased from ~135MB without embeddings)
  - **Configuration**: `enable_sample_cache: true` (default), `sample_cache_version: 3`
  - **Key benefit**: Eliminates CLIP inference on cache hits (saves ~150-700ms per sample)
  - **Backward compatibility**: Gracefully handles v2 cache (recomputes embeddings if missing)

### November 5, 2025 - Reference Embedding Cache Fix
- **Cache validation bug fixed**: Reference embeddings now load correctly from cache
  - **Issue**: numpy saves dicts as 0-dim arrays, validation failed on `isinstance(ref_cache, dict)`
  - **Fix**: Added `.item()` extraction for numpy scalar arrays before validation
  - **Impact**: Saves 2-5 minutes per retrieval by reusing cached reference embeddings

### November 5, 2025 - Cross-Dataset Sample Caching (v2)
- **Sample metadata caching**: Cross-dataset retrieval optimization system
  - Caches visual/semantic metrics and EfficientNet predictions
  - Reusable across multiple dataset retrievals from the same source
  - **Cache location**: `cache_base_dir/sample_metadata_cache/{hash[:2]}/sample_{hash}_v{version}.json`
  - **Test coverage**: 16 comprehensive tests in `tests/test_sample_cache.py`

### November 19, 2025 - Caltech101 Torchvision Compatibility Fix
- **Class list mismatch fixed**: Aligned class_names with torchvision's actual implementation
  - **Issue**: Torchvision explicitly removes 'BACKGROUND_Google' category, leaving 101 classes (labels 0-100)
  - **Root cause**: MAVERIC had 102 classes including 'background_google', causing index mismatch
  - **Impact**: 'yin_yang' class showed 0 samples (index 101 doesn't exist in torchvision)
  - **Fix**: Updated class_names to match torchvision's 101 sorted categories (excluding BACKGROUND_Google)
  - **Result**: All 101 Caltech101 classes now correctly mapped to torchvision labels
  - **Location**: [elevater_datasets.py](maveric/datasets/elevater_datasets.py)

### November 18, 2025 - Caltech101 Dataset Fixes
- **Missing "leopards" class**: Added missing class to Caltech101 dataset
  - **Issue**: "leopards" class was completely missing from class names list
  - **Fix**: Added "leopards" at correct position in alphabetical ordering
  - **Location**: [elevater_datasets.py](maveric/datasets/elevater_datasets.py)

### November 13, 2025 - Statistics Display Improvements
- **Retrieval statistics fix**: Enhanced progress display for better clarity
  - Always shows cache hits and downloads (even if 0) for consistency
  - Improved verification: Processed = Cache Hits + Downloads
  - Better batch position tracking and index information display
  - **Location**: [progress.py](maveric/core/progress.py)

### November 2, 2025 - Critical Bug Fix
- **Class name extraction bug**: Fixed GTSRB showing only 3/43 classes due to underscore parsing issue
  - All datasets with underscores in class names now work correctly (e.g., `ahead_only`, `beware_of_ice_snow`)
  - Test script included: `test_class_name_extraction.py`

### October 30, 2025 - Major Performance & Reliability Updates

**Critical Configuration Changes**:
- `enable_target_class_quality`: **Default changed to `false`** (was `true`) - provides 50-70% faster retrieval

**New Features**:
- **Atomic file writes**: `save_json_atomic()` in `io_utils.py` prevents corruption on Google Drive/NFS
- **Enhanced cache validation**: Automatic detection and regeneration of corrupted cache files
- **Diagnostic logging**: Comprehensive file-based dataset debugging with directory structure analysis
- **Progress logging**: Long operations now show progress (CLIP loading, dataset loading, reference generation)

**Documentation**:
- New `docs/bugfixes/` directory with 88 KB of comprehensive bug fix documentation
- Complete retrieval analysis and performance optimization guides
- 10 complete CIFAR-100 experiment runs documented in `experiments/CIFAR100_Experiments.txt`

**Performance Impact**:
- 50-70% faster data retrieval (EfficientNet disabled by default)
- No more file corruption or hanging on network filesystems
- Better debugging capabilities with enhanced logging

See the "Recent Improvements" section below for detailed information.

## Architecture Overview

MAVERIC is a multi-modal dataset curation system for vision-language models. The codebase follows a modular architecture.

**Visual Architecture**: See [docs/maveric_pipeline.svg](docs/maveric_pipeline.svg) for a comprehensive diagram of the complete pipeline including reference generation.

**Key Components**:

- **`maveric/main.py`**: Main MAVERIC class providing high-level API for retrieval, quality control, and model customization
- **`maveric/config.py`**: Dataclass-based configuration system (MAVERICConfig, TrainingConfig, ExperimentConfig)
- **`maveric/core/`**: Base interfaces, exceptions, abstract components, and progress tracking system
  - `base.py`: BaseComponent, BaseDataset, BaseMetric abstract classes
  - `interfaces.py`: RetrievalResult, QualityResult, CustomizationResult, ProgressCallback
  - `exceptions.py`: MAVERICError hierarchy (ConfigurationError, DatasetError, ModelError, CacheError)
  - `progress.py`: RealTimeStats for live statistics tracking
- **`maveric/retrieval/`**: Dataset retrieval and caching system with CLIP-based embedding
  - `retriever.py`: Main retrieval engine with quality metric computation
  - `cache_manager.py`: Smart caching for images, embeddings, and results
  - `sample_cache_manager.py`: **NEW** Cross-dataset sample metadata caching
  - `dataset_handlers.py`: Handlers for different dataset formats (REACT, etc.)
- **`maveric/quality/`**: Quality assessment metrics (visual, semantic, multimodal consistency)
  - `quality_controller.py`: Main quality control orchestration
  - `filters.py`: Quality-based filtering and dataset balancing
  - `metrics/`: Quality metric implementations organized by category
- **`maveric/customization/`**: Model fine-tuning with filtered data
  - `model_customizer.py`: High-level fine-tuning API
  - `training.py`: Training loop with Trainer and TrainingMonitor
  - `evaluation.py`: Model evaluation on test sets
- **`maveric/visualization/`**: Comprehensive visualization and interactive GUI system
  - `distributions.py`: MetricsVisualizer for distribution plots
  - `samples.py`: SampleVisualizer for image galleries
  - `interactive.py`: Full-featured MAVERICInteractiveQualityControl GUI (primary interactive interface)
  - `plots.py`: Utility plotting functions
  - **Note**: The `maveric/interactive/` folder has been removed (redundant components)
- **`maveric/datasets/`**: Unified ELEVATER benchmark dataset handler (official 20 datasets: 9 torchvision + 11 file-based)
  - `elevater_datasets.py`: ELEVATER benchmark dataset implementations with REACT-style text templates
  - `dataset_factory.py`: Factory for creating dataset instances
- **`maveric/models/`**: CLIP model wrappers and factory patterns
  - `clip_wrapper.py`: CLIP model wrapper with utilities
  - `model_factory.py`: Factory for creating model instances
- **`maveric/utils/`**: Command-line interface, I/O utilities, logging, and visualization helpers
  - `cli.py`: Complete CLI for all MAVERIC operations
  - `io_utils.py`: File handling and data serialization
  - `logging.py`: Structured logging system
  - `visualization.py`: Visualization helper utilities

## Development Commands

### Installation
```bash
# Development install
pip install -e ".[dev]"

# Install with docs dependencies
pip install -e ".[dev,docs]"
```

### Testing

#### Basic Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=maveric --cov-report=html

# Run specific test file
pytest tests/test_quality_metrics.py

# Run specific test
pytest tests/test_main.py::test_retrieve
```

#### Headless Environment Testing
For Docker, CI/CD, or remote environments without display:

```bash
# Set matplotlib backend and run tests
MPLBACKEND=Agg pytest

# Alternative: set in environment
export MPLBACKEND=Agg
pytest
```

#### Common Test Issues and Solutions

**Import Errors:**
- Missing `openai-clip`: `pip install openai-clip`
- Missing `torch`/`torchvision`: Install appropriate version for your system
- Missing `sentence-transformers`: `pip install sentence-transformers` (required for target class quality metrics)
- Missing `scikit-learn`: `pip install scikit-learn` (required for cosine similarity calculations)
- Missing `langdetect`: `pip install langdetect` (required for text quality metrics)
- Missing `torchvision`: Required for EfficientNet-B0 in semantic quality assessment
- `libGL.so.1` errors: Use `opencv-python-headless` instead of `opencv-python`

**Matplotlib/Visualization Issues:**
- `seaborn` style not found: Updated to use 'default' style
- Display errors in headless environments: Set `MPLBACKEND=Agg`

**PIL API Changes:**
- `Image.BLUR` deprecated: Use `ImageFilter.BLUR` instead
- Import `ImageFilter` from PIL for filtering operations

**Test Environment Issues:**
- Use `tmp_path` fixture for temporary directories
- Mock external dependencies properly
- Import all required modules in test files (`torch`, `numpy`, etc.)

### Code Quality
```bash
# Format code
black maveric/ tests/

# Lint
flake8 maveric/ tests/

# Type checking
mypy maveric/
```

### CLI Usage
The package provides a CLI tool accessible via `maveric` command:

```bash
# Retrieve samples (default: without EfficientNet for 50-70% faster retrieval)
python experiments/01_data_retrieval.py --config config.yaml

# Retrieve samples WITH EfficientNet (includes Class_*_efficientNet_score fields)
python experiments/01_data_retrieval.py --config config.yaml --enable-efficientnet

# Apply quality control
maveric quality-control --input results.json --thresholds thresholds.json --output filtered.json

# Customize model
maveric customize --input filtered.json --model openai/clip-vit-base-patch32 --epochs 10 --output-dir ./models

# Visualize distributions
maveric visualize --input results.json --output-dir ./plots
```

## Experiment Scripts

The `experiments/` directory contains end-to-end workflows for different stages:

- **`00_setup.py`**: Environment setup script for automated installation and configuration
- **`01_data_retrieval.py`**: Data retrieval from source datasets with CLIP-based matching
- **`02_data_curation.py`**: Quality control and filtering of retrieved data
- **`03_model_customization.py`**: Model fine-tuning with curated datasets
- **`04_results_analysis.py`**: Analysis and visualization of experiment results
- **`maveric_config.yaml`**: Configuration file with optimal hyperparameters

Each script is designed to be run independently or as part of a complete pipeline.

## Configuration System

MAVERIC uses dataclass-based configuration in `config.py`:

- **MAVERICConfig**: Main system configuration (models, caching, quality thresholds, progress display)
- **TrainingConfig**: Model training parameters with regularization and augmentation
- **ExperimentConfig**: Experiment management and tracking

Key configuration options:
- `enable_real_time_stats`: Show live download/cache statistics during retrieval (default: true)
- `enable_target_class_quality`: Enable EfficientNet-based TargetClassQualityMetric (default: **false** for 50-70% faster retrieval, set to true for comprehensive quality assessment)
- `request_timeout`: HTTP request timeout in seconds (default: 5)
- `clip_model`: CLIP model to use (default: "ViT-B/32")
- `cache_base_dir`: Directory for caching downloaded images and results
- `batch_size`: Processing batch size
- `retrieval_rotation_size`: Samples per file when saving results and training data (default: 1000)
- `quality_metrics`: List of quality metrics to compute (default: visual: resolution, sharpness, color_diversity; semantic: text_quality, caption_length; multimodal: target_class_quality, multimodal_consistency)
- `metric_weights`: Weights for composite scoring across modalities (img2img: 0.4, txt2txt/img2txt/txt2img: 0.2 each)
- `seed`: Random seed for reproducible sampling (default: 42)

TrainingConfig key parameters:
- `epochs`: Number of training epochs (default: 20, updated Nov 2025 for better convergence)
- `learning_rate`: Learning rate for optimizer (default: 0.0000007, tuned for CLIP fine-tuning)
- `weight_decay`: L2 regularization weight (default: 0.07, increased for better generalization)
- `gradient_clip_value`: Gradient clipping threshold (default: 0.75, prevents training instability)
- `use_regularization`: Enable MSE regularization to prevent catastrophic forgetting (default: true)
- `regularization_weight`: Weight for regularization loss (default: 0.75, balanced to preserve original model knowledge)
- `use_augmentation`: Enable RandAugment data augmentation (default: true)
- `augmentation_strength`: RandAugment num_ops parameter (default: 7, more diverse augmentations)
- `augmentation_magnitude`: RandAugment magnitude parameter (default: 22, stronger transformations)
- `optimizer`: Optimizer type - adamw, adam, or sgd (default: "adamw")
- `scheduler`: Learning rate scheduler - cosine, linear, or constant (default: "cosine")
- `use_validation`: Enable validation split during training (default: true)
- `validation_method`: Validation strategy - stratified_kfold or simple_split (default: "stratified_kfold")

**Note**: Default values shown above reflect the optimized hyperparameters in `experiments/maveric_config.yaml` (updated Nov 20, 2025) based on extensive CIFAR-100 experiments. These settings provide a good balance between model adaptation and preservation of original CLIP capabilities.

Configuration can be loaded from YAML/JSON files:
```python
config = MAVERICConfig.from_yaml('config.yaml')
maveric = MAVERIC.from_config_file('config.yaml')
```

## Key Components

### Quality Metrics (`maveric/quality/metrics/`)
**MAVERIC now properly implements all three metric categories:**

- **Visual metrics** (`visual_metrics.py`): Image-only quality assessment
  - `ResolutionMetric`: Image resolution evaluation  
  - `SharpnessMetric`: Laplacian variance-based sharpness
  - `ColorDiversityMetric`: Color channel standard deviation

- **Semantic metrics** (`semantic_metrics.py`): Text-only quality assessment ✅ **NOW ENABLED**
  - `TextQualityMetric`: Caption quality (length, vocabulary, language detection)
  - `CaptionLengthMetric`: Caption length appropriateness

- **Multimodal metrics** (`multimodal_metrics.py`): Cross-modal quality assessment
  - `MultimodalConsistencyMetric`: CLIP-based cross-modal alignment
  - `CrossModalAlignmentMetric`: Direct image-text similarity  
  - `TargetClassQualityMetric`: EfficientNet + CLIP for per-class quality assessment

### Core System (`maveric/core/`)
- **Base Classes**: Abstract components for datasets, metrics, and system components
- **Interfaces**: Result types and callback interfaces for consistent API
- **Exception Handling**: Centralized exception hierarchy for error management
- **Progress Tracking**: `RealTimeStats` system for live download/cache statistics display

### Retrieval System (`maveric/retrieval/`)
- CLIP-based embedding similarity matching with per-class quality assessment
- Smart caching system for images and embeddings
- Dataset handlers for different source formats
- **NEW**: Class mapping visualization showing target dataset classes → ImageNet-1K mappings
- **NEW**: Progress bar suppression for cleaner console output

### Interactive Dashboard (`maveric/interactive/`)
- **ThresholdSelector** (`threshold_selector.py`): Jupyter widget for real-time threshold tuning with metric weight controls
- **QualityDashboard** (`quality_dashboard.py`): Quality distribution visualization (updated to exclude global composite_quality, now uses imagenet_probability)
- **Widgets** (`widgets.py`): Reusable UI components including threshold sliders, weight controls, and metric selectors
- Sample gallery with filtering
- Interactive controls for metric weights with auto-normalizing sliders that maintain 1.0 sum

### Customization System (`maveric/customization/`)
- **ModelCustomizer** (`model_customizer.py`): High-level API for model fine-tuning with regularization
  - **Class name normalization**: Intelligently matches dataset class names with template placeholders
  - **Dataset-specific templates**: Automatically retrieves REACT-style text templates for evaluation
- **Trainer** (`training.py`): Training loop implementation with validation and monitoring
- **TrainingMonitor** (`training.py`): Real-time training metrics tracking and logging
- **Evaluator** (`evaluation.py`): Model evaluation on test sets with comprehensive metrics
  - **Template integration**: Uses dataset-specific prompts for consistent evaluation

### Visualization System (`maveric/visualization/`)
- **MetricsVisualizer** (`distributions.py`): Quality metric distribution plots and analysis
- **SampleVisualizer** (`samples.py`): Image galleries and sample inspection tools
- **Interactive GUI** (`interactive.py`): Full-featured interactive quality control interface for Jupyter/Colab
  - `MAVERICInteractiveQualityControl`: Main interactive controller class
  - `create_quality_control()`: Factory function for creating GUI instances
  - `start_interactive_gui()`: Convenience function for launching GUI
  - Automatic fallback when ipywidgets unavailable
- **Plotting Utilities** (`plots.py`): Class distribution, correlation matrices, quality comparisons

### Utilities (`maveric/utils/`)
- **CLI System** (`cli.py`): Complete command-line interface for all MAVERIC operations (retrieve, quality-control, customize, visualize)
- **I/O Utilities** (`io_utils.py`): File handling, data serialization, and configuration management
  - **NEW**: `save_json_atomic()`: Atomic write pattern for network filesystems (prevents corruption on Google Drive/NFS)
- **Logging** (`logging.py`): Structured logging system with configurable levels and formatters
- **Visualization Helpers** (`visualization.py`): Utility functions for plotting and data visualization

## Data Flow

**Visual Diagram**: See [docs/maveric_pipeline.svg](docs/maveric_pipeline.svg) for a complete visualization of the 4-stage pipeline.

**Pipeline Stages**:
1. **Retrieval**: Load source dataset → Generate reference embeddings (one-time) → Generate CLIP embeddings → Match against target dataset embeddings
2. **Quality Assessment**: Apply visual/semantic metrics → Score each sample → Calculate composite quality scores per class
3. **Filtering**: Apply thresholds → Balance dataset → Export filtered results
4. **Customization**: Fine-tune model on filtered data → Evaluate performance

**Reference Generation**: See [docs/REFERENCE_GENERATION.md](docs/REFERENCE_GENERATION.md) for detailed documentation on the reference generation subsystem (sample selection, REACT templates, CLIP embeddings, and caching).

### Data Formats and Outputs

**Retrieval Results**: Saved as rotation files (JSON/pickle) with hierarchical structure:
- Image metadata (URL, caption, dimensions)
- Visual metrics: `resolution_score`, `sharpness_score`, `color_score`
- Semantic metrics: `text_quality_score`, `caption_length_score`
- Multimodal metrics: Per-class scores (e.g., `Class_airplane_img2img`, `Class_airplane_efficientNet_score`)
- Composite scores: `weighted_class_score`, `consistency` per class

**Curated Training Data**: Hierarchical directory structure to avoid NFS issues:
```
training_data/
├── class_000/
│   ├── batch_000/
│   │   ├── sample_000.jpg
│   │   ├── sample_001.jpg
│   │   └── ...
│   └── batch_001/
│       └── ...
└── class_001/
    └── ...
```

**Model Checkpoints**: PyTorch model files with training state:
- `best_model.pth`: Best model based on validation accuracy
- Training configuration and hyperparameters embedded
- Compatible with CLIP architecture for downstream tasks

### Advanced Quality Assessment

**Per-Class Target Quality Scoring**: **MAJOR UPDATE** - Target class quality is now calculated per-class instead of globally, following the same pattern as `hybrid_score` and `consistency`:
- For each target dataset class (e.g., CIFAR-10's 10 classes), a class-specific composite quality score is computed
- Uses EfficientNet-B0 + CLIP embeddings for semantic similarity with ImageNet classes
- Results in `Class_{class_name}_efficientNet_score` columns (e.g., `Class_airplane_efficientNet_score`) based on CLIP similarity with predicted ImageNet class
- Enables class-aware quality assessment - images are evaluated specifically for how well they represent each class
- Class selection combines similarity score with class-specific quality score using configurable weights

**Target Class Quality Metric**: **Properly categorized as multimodal** - Located in `multimodal_metrics.py` and used for per-class quality assessment:
- Uses EfficientNet-B0 (CPU-only) for universal image classification
- Employs CLIP for semantic similarity with ImageNet classes (more robust than sentence transformers)
- Pre-computes CLIP embeddings for ImageNet classes for efficiency
- Focuses on semantically relevant ImageNet classes based on caption content
- Works universally across all ELEVATER datasets without manual class mappings
- **Current implementation**: Returns CLIP similarity × ImageNet probability as final score
- **OPTIMIZED**: Batch processing computes mappings for all target classes using single EfficientNet inference
- Provides comprehensive quality scores considering both visual quality and semantic relevance

**Semantic Quality Filtering**: **NEW** - Pure text quality assessment now enabled by default:
- Text quality metrics filter poor captions (wrong language, too short/long, low vocabulary diversity)
- Caption length metrics ensure appropriate caption sizes
- Semantic filtering works alongside visual and multimodal quality assessment

**Class Selection Architecture**: Simplified class selection logic using weighted similarity scores:
- Weighted similarity scoring: Uses img2img, txt2txt, img2txt, txt2img metrics with configurable weights
- Pure similarity-based approach: Class selection based on weighted_class_score only
- Interactive controls: Real-time metric weight adjustment via Jupyter widgets

## Testing Strategy

- Unit tests for each component in `tests/`
- Configuration fixtures in `tests/conftest.py`
- Quality metrics validation with known datasets
- Integration tests for full pipeline workflows

## Cache Management

Images, embeddings, and reference data are cached in configurable directories:
- Image cache: JPEG compressed images for fast loading
- **Sample metadata cache**: **NEW** Cross-dataset sample caching for 60-85% speedup
- Embedding cache: Precomputed CLIP embeddings
- Results cache: Serialized retrieval and quality results
- Reference images cache: Reference images used for embedding generation (organized by dataset/class)
- Reference texts cache: Text templates and generated prompts for verification

### Cross-Dataset Sample Caching (v3 - UPDATED)

**Purpose**: Cache reusable data across multiple dataset retrievals to dramatically reduce processing time.

**What's Cached** (per sample - v3):
- Visual metrics (resolution, sharpness, color_diversity)
- Semantic metrics (text_quality, caption_length)
- **CLIP embeddings (image + text)** ⭐ NEW in v3 - base64 encoded
- EfficientNet predictions (ImageNet class + probability)

**What's NOT Cached**:
- Per-class similarity scores (`Class_{name}_img2img`, `Class_{name}_txt2txt`, etc.) - dataset-specific
- Class-specific quality scores - dataset-specific
- Dataset-specific reference comparisons

**Performance Impact**:
```
Cache v2 (without CLIP embeddings):
  First retrieval (CIFAR-10, 10k samples):    ~2.2 hours (builds cache)
  Second retrieval (CIFAR-100, 10k samples):  ~0.5 hours (75% faster)

Cache v3 (WITH CLIP embeddings):  ⭐ NEW
  First retrieval (CIFAR-10, 10k samples):    ~2.2 hours (builds cache)
  Second retrieval (CIFAR-100, 10k samples):  ~0.3 hours (85% faster!)
  Eliminates CLIP inference: saves 150-700ms per sample on cache hits

Total for 20 datasets: ~7.0 hrs vs 44.4 hrs (84% savings!)
```

**Storage**:
- **Per sample (v3)**: ~17KB (~500 bytes metrics + ~16KB embeddings)
- **270K samples**: ~4.5GB (trade-off: more storage for faster retrieval)
- **Per sample (v2)**: ~500 bytes (no embeddings)
- **270K samples**: ~135MB (v2 - less storage, slower retrieval)

**Configuration**:
```yaml
enable_sample_cache: true          # Enable/disable caching (default: true)
sample_cache_version: 3            # Cache format version (v3: includes CLIP embeddings)
```

**Version History**:
- **v3** (current): Caches CLIP embeddings + metrics + EfficientNet predictions
- **v2**: Caches metrics + EfficientNet predictions only (CLIP computed from cached images)
- **v1**: Initial implementation

**Cache Invalidation**:
- Increment `sample_cache_version` in config when metric computation changes
- Clear specific URL: `cache_manager.clear_cache(url="...")`
- Clear all: `cache_manager.clear_cache()`

### Cache Directory Structure
```
maveric_cache/
├── image_cache/                   # Cached downloaded images
│   └── {hash[:2]}/
│       └── img_{hash}.jpg
├── sample_metadata_cache/         # ⭐ Cross-dataset sample cache (v3: includes CLIP embeddings)
│   └── {hash[:2]}/
│       └── sample_{hash}_v3.json
├── reference_images/              # Reference samples per dataset
│   └── {dataset_name}/
│       └── {class_name}/
│           ├── ref_000.jpg
│           └── ...
├── reference_texts/               # Text templates per dataset
│   └── {dataset_name}_texts.json
└── embeddings/                    # Dataset-specific reference embeddings
    └── {dataset}_reference_embeddings.npz
```

### Sample Cache JSON Format (v3)
```json
{
  "cache_version": 3,
  "url": "https://...",
  "url_hash": "a1b2c3d4...",
  "text": "A photo of a cat",
  "last_updated": "2025-11-05T10:30:00Z",
  "visual_metrics": {
    "resolution_score": 0.895,
    "sharpness_score": 0.923,
    "color_score": 0.812
  },
  "semantic_metrics": {
    "text_quality_score": 0.850,
    "caption_length_score": 0.920
  },
  "clip_embeddings": {
    "image_embedding": "base64_encoded_string_of_numpy_array...",
    "text_embedding": "base64_encoded_string_of_numpy_array...",
    "image_shape": [1, 512],
    "text_shape": [1, 512],
    "dtype": "float32"
  },
  "efficientnet_predictions": {
    "imagenet_predicted_class": "tabby cat",
    "imagenet_probability": 0.892
  }
}
```

**Note**: v3 caches CLIP embeddings as base64-encoded numpy arrays. This increases file size (~17KB vs ~500 bytes) but eliminates CLIP inference on cache hits (saves 150-700ms per sample).

Reference texts files contain:
- `templates`: Original text templates used
- `class_names`: List of all class names in the dataset
- `generated_prompts`: Dictionary mapping each class to its generated prompts

## Dataset Support

MAVERIC supports all 20 official ELEVATER benchmark datasets through a unified handler:

### Torchvision-based Datasets (7) - **Updated February 10, 2026**:
- CIFAR-10, CIFAR-100
- Country211, EuroSAT
- GTSRB, Oxford Flowers102, Oxford Pets

**Note**: Food101 and FGVCAircraft were migrated to file-based on February 10, 2026. Caltech101 was already file-based.

### File-based Datasets (13) - **Updated February 10, 2026**:
- Caltech101, DTD, FER2013
- **FGVCAircraft** ⬅️ (migrated from torchvision February 10, 2026)
- **Food101** ⬅️ (migrated from torchvision February 10, 2026)
- Hateful Memes, KITTI Distance, MNIST
- PatchCamelyon, RenderedSST2, RESISC45
- Stanford Cars, VOC2007

**Important**: File-based datasets require manual test data download and organization. See [FILE_BASED_DATASETS_GUIDE.md](FILE_BASED_DATASETS_GUIDE.md) for setup instructions.

**Benefits of Torchvision datasets**: Automatic downloading, standardized interfaces, and optimized loading.

### REACT-Style Text Templates

**NEW**: MAVERIC now implements dataset-specific text templates following the REACT benchmark pattern:

**Purpose**: Provide contextually appropriate prompts for each dataset to improve CLIP-based retrieval quality.

**Implementation**: The `get_text_templates()` method in `elevater_datasets.py` returns multiple templates per dataset with `{}` placeholders for class names.

**Example Templates**:
```python
# DTD (textures)
"a photo of a {} texture."
"a close-up photo of a {} texture."

# EuroSAT (satellite imagery)
"a centered satellite photo of {}."
"a satellite photo of {}."

# GTSRB (traffic signs)
"a zoomed in photo of a {} traffic sign."
"a centered photo of a {} traffic sign."

# Food101 (food items)
"a photo of {}, a type of food."
"a photo of {} food."

# Oxford Flowers102 (flowers)
"a photo of a {}, a type of flower."
"a close-up photo of a {} flower."
```

**Coverage**: Custom templates for 15+ datasets including DTD, EuroSAT, FER2013, Food101, GTSRB, Oxford Flowers102, Oxford Pets, CIFAR-10, CIFAR-100, Caltech101, Country211, FGVCAircraft, MNIST, RenderedSST2, and Stanford Cars.

**Default Fallback**: For datasets without custom templates, uses generic prompts:
- "a photo of a {}."
- "a picture of a {}."
- "an image of a {}."

**Integration**: Templates are automatically used during:
- Model evaluation in `evaluation.py`
- Model customization in `model_customizer.py`
- Reference text generation for retrieval

**Best Practices**:
1. **Template Consistency**: Always use the same templates for retrieval and evaluation to maintain consistency
2. **Class Name Formatting**: The system automatically normalizes class names (e.g., "speed_limit_30" → "speed limit 30") to match template expectations
3. **Custom Templates**: To add templates for new datasets, update the `dataset_templates` dictionary in `get_text_templates()`
4. **Template Testing**: Verify templates produce meaningful prompts by checking generated reference texts in cache

### Caltech101 Special Notes

**Torchvision Behavior**: Torchvision's Caltech101 implementation automatically:
- Removes the 'BACKGROUND_Google' category
- Sorts remaining 101 categories alphabetically
- Assigns labels 0-100 to these categories

**Manual Download Workaround** (if torchvision download URLs are broken):
1. Download Caltech101 from Kaggle: https://www.kaggle.com/datasets/imbikramsaha/caltech-101
2. Place in: `{cache_dir}/datasets/caltech101/101_ObjectCategories/`
3. Torchvision will use the local files automatically
4. Note: You'll get 8,677 samples (torchvision excludes BACKGROUND_Google images)

**Expected Behavior**:
- Dataset size: 8,677 samples (not 9,144 - BACKGROUND_Google excluded)
- Number of classes: 101 (not 102)
- All classes including 'yin_yang' will have samples

### Important Notes for Large Datasets

**Food101 and Other Large Datasets**: Some torchvision datasets (particularly Food101, ~5GB with 75,750 training samples) require special handling:
- **First-time setup**: Automatic download via torchvision (one-time, ~5GB download)
- **Optimized class indexing**: Single-pass scan through dataset to build class-to-sample mapping
  - Original approach: O(n × c) = 101 classes × 75,750 samples = 7.6M iterations
  - Optimized approach: O(n) = 75,750 samples with progress updates
- **Progress logging**: Real-time progress updates during index building (20 checkpoints)
- **Google Drive considerations**: Initial scan may be slower on Google Drive vs. local storage

**What You'll See During Reference Sample Selection**:
```
Selecting FOOD101 sample data randomly...
  Dataset size: 75,750 samples
  Number of classes: 101
  Samples per class: 10
  Building class index map (one-time scan)...
    Progress: 3,787/75,750 (5.0%)
    Progress: 7,575/75,750 (10.0%)
    ...
    Progress: 75,750/75,750 (100.0%)
  ✅ Index map built. Processing classes...
  [1/101] Class 'apple_pie': 750 samples
  [2/101] Class 'baby_back_ribs': 750 samples
  ...
✅ Reference sampling complete: 101 classes, 1010 total images
```

**Performance Notes**:
- Index building takes ~2-5 minutes on Google Drive, ~30 seconds on local SSD
- Subsequent runs use cached embeddings (no need to rebuild)
- Other large datasets (Caltech101, Country211) benefit from same optimization

## Important Development Notes

### Package Structure
- Entry point: `setup.py` defines package metadata and dependencies
- Core package: `maveric/` contains all source code (~12,800 lines across 45+ files)
- Tests: `tests/` contains unit and integration tests (9 test files)
  - `test_sample_cache.py`: 16 comprehensive cross-dataset caching tests
  - `test_optimization.py`: Validates EfficientNet batch processing optimizations
  - `test_class_name_extraction.py`: Validates class name parsing for datasets with underscores
- Examples: `examples/` contains usage examples
  - `interactive_notebook.ipynb`: Interactive Jupyter notebook demonstrating MAVERIC features
- Experiments: `experiments/` contains end-to-end workflow scripts
  - `CIFAR100_Experiments.txt`: 10 complete experiment runs with manual hyperparameter tuning results (362 lines)
  - `maveric_config.yaml`: Updated configuration with optimized hyperparameters (Nov 20, 2025)
- Documentation: Comprehensive documentation suite (~150 KB total)
  - `README.md`: Main project documentation (16.6 KB)
  - `CLAUDE.md`: Developer guide for Claude Code (this file, 48 KB, 929 lines)
  - `CODEBASE_ANALYSIS.md`: Architecture and extension opportunities (15 KB)
  - `docs/bugfixes/`: Bug fix documentation suite (88 KB total, 8 files)
  - `docs/CROSS_DATASET_CACHING.md`: Cross-dataset sample caching guide (20 KB)
  - `docs/DATASET_DOWNLOAD_ISSUES.md`: Dataset download troubleshooting (9.5 KB)
  - `docs/maveric-api-docs.md`: API reference documentation (12 KB)
  - `docs/detailed_documentation.txt`: Detailed API and architecture docs (16 KB)

### CLI Entry Point
The CLI entry point is correctly defined in `setup.py:49` as `maveric=maveric.utils.cli:main`, which points to the actual CLI implementation in `maveric/utils/cli.py`.

### Testing Environment Setup
- All tests automatically force CPU device via `conftest.py` device fixture
- Random seeds are set for reproducibility (numpy=42, torch=42)
- No pytest configuration files - uses defaults
- For headless environments (Docker/CI): set `MPLBACKEND=Agg` before running tests

### Development Dependencies Structure
The project uses layered requirements:
- `requirements.txt`: Core runtime dependencies only
- `requirements-dev.txt`: Includes base requirements via `-r requirements.txt` plus dev tools

### Configuration Architecture Details
Configuration uses dataclasses in `config.py` with three main classes:
- `MAVERICConfig`: System config with intelligent defaults (auto device detection, directory creation)
- `TrainingConfig`: Model training parameters
- `ExperimentConfig`: Experiment management and tracking

Key config features:
- YAML/JSON loading support
- Auto device detection when set to "auto"
- Automatic directory creation in `__post_init__`
- Smart path handling: Replaces inaccessible `/content/` paths with local alternatives when not in Colab
- Legacy field mapping: Automatically maps deprecated config fields to current versions

### Google Colab and Drive Compatibility
MAVERIC is optimized for Google Colab environments:
- **Setup script**: `experiments/00_setup.py` handles Colab-specific setup including Drive mounting
- **Path handling**: Config system detects and replaces inaccessible Colab paths
- **Hierarchical storage**: Avoids Google Drive NFS mount issues with hierarchical file organization
- **Progress tracking**: Configurable progress displays suitable for Colab notebooks
- **Environment variables**: Automatic setup of `MAVERIC_BASE_DIR`, `MAVERIC_CACHE_DIR`, etc.
- **System dependencies**: Automated installation of required system packages

## Performance & Architecture Improvements

### CPU-Only Data Retrieval
- **TargetClassQualityMetric** now uses EfficientNet-B0 on CPU during data retrieval
- Eliminates GPU memory usage during the data collection phase
- Maintains high-quality assessment while reducing hardware requirements

### EfficientNet Quality Metrics (Disabled by Default)
**Default Behavior**: EfficientNet-based quality metrics are **DISABLED by default** for 50-70% faster data retrieval.

**To enable EfficientNet calculations**:

**Command-line flag**:
```bash
python experiments/01_data_retrieval.py --config config.yaml --enable-efficientnet
```

**Configuration file**:
```yaml
enable_target_class_quality: true  # Enable EfficientNet calculations
```

**What gets computed when enabled**:
- `Class_{class_name}_efficientNet_score` fields for per-class quality assessment
- `Class_{class_name}_clip_similarity_to_imagenet` fields for ImageNet alignment
- `imagenet_predicted_class` and `imagenet_probability` fields
- Full EfficientNet-B0 model loading and inference

**Performance impact**:
- **Default (disabled)**: ~50-70% faster data retrieval
- **Enabled**: More comprehensive quality metrics but slower processing
- All other quality metrics (visual, semantic, similarity-based) are always computed

**When to enable**:
- Need per-class ImageNet-based quality assessment
- Filtering based on EfficientNet scores
- Final production data curation with comprehensive metrics

**When to keep disabled (default)**:
- Initial data exploration when you want quick results
- EfficientNet scores not needed for your filtering criteria
- Limited computational resources or time constraints
- Working with very large datasets (>100k samples)

### Data Curation Compatibility

The `02_data_curation.py` script **automatically handles both types of data**:

**Without EfficientNet metrics** (default retrieval):
- Script automatically detects missing EfficientNet fields
- Filters skip missing metrics gracefully (no errors)
- All other thresholds (visual, semantic, similarity) are still applied
- Quality control works identically, just with fewer metrics

**With EfficientNet metrics** (`--enable-efficientnet` retrieval):
- All quality thresholds are applied, including EfficientNet-based ones
- Full range of filtering options available

**Example workflow**:
```bash
# Step 1: Fast retrieval without EfficientNet (default)
python experiments/01_data_retrieval.py --config config.yaml

# Step 2: Curation works automatically (no special flags needed)
python experiments/02_data_curation.py --input-dir results/cifar10/raw --dataset-name cifar10 --config config.yaml
```

The curation script will display:
```
ℹ️  EfficientNet metrics not present (default behavior for faster retrieval)
   Visual, semantic, and similarity metrics are still available for filtering
```

**Note**: If your quality thresholds include EfficientNet-based metrics (like `imagenet_probability`), they will be automatically skipped without causing errors.

### Optimized Quality Score Calculation
- **Batch EfficientNet Processing**: EfficientNet inference runs only once per image, not once per target class
- **Probability Reuse**: Same ImageNet probabilities are reused for all target class mappings
- **Performance Improvement**: Reduces computational overhead from O(N) to O(1) EfficientNet calls per image (where N = number of target classes)
- **Memory Efficiency**: Computes all ImageNet mappings from a single probability tensor

### Model Customization with Regularization
- **Locked-Text Tuning**: Only vision encoder is fine-tuned, text encoder remains frozen
- **MSE Regularization**: Prevents catastrophic forgetting by maintaining similarity to original vision weights
- **Configurable Regularization Weight**: Control trade-off between adaptation and preservation (default: 0.5)
- **Formula**: `total_loss = task_loss + regularization_weight × MSE(current_weights, original_weights)`
- **Performance**: Optimal `regularization_weight` typically in range [0.4, 0.6] based on empirical results

### Progress Bar Management  
- **Real-time statistics**: Set `enable_real_time_stats: false` in config to disable live download/cache statistics
- **Console-friendly**: Configurable progress display for production environments

### Per-Class Quality Architecture
- **Eliminated global quality scores**: No more single `composite_quality` per sample - now using per-class `imagenet_probability`
- **Class-specific assessment**: Quality evaluated relative to each target class using CLIP-ImageNet mappings
- **Consistent data structure**: Follows same pattern as similarity metrics (`Class_{name}_{metric}`)
- **Enhanced class selection**: Combines similarity and quality at the class level with configurable weighting

### Memory Optimization
- **Efficient caching**: Smart image and embedding cache management
- **Rotation files**: Large datasets automatically split into manageable chunks (default: 1000 samples per file)
- **Hierarchical file structure**: Implements hierarchical organization to avoid Google Drive NFS mount errors
- **Resource management**: Better GPU/CPU resource allocation during different phases

### Recent Improvements (Latest Commits)

**November 21, 2025 - Critical Evaluation Fixes**:
- **Class Name Capitalization Bug Fix** (~4-5% accuracy improvement):
  - **Problem**: Training JSON had lowercase labels, AND torchvision dynamically generates class names that differ from REACT
  - **Impact**: Oxford Pets - 82-83% (lowercase) → 87%+ (proper REACT class names)
  - **Root cause**: Loading class names from dataset handler allowed torchvision to override with its own Title Case names (e.g., "American Bulldog") instead of using REACT's mixed-case format (e.g., "american bulldog")
  - **Fix**: Load class names DIRECTLY from `ELEVATER_DATASETS` dictionary, not from dataset handler
  - **Key insight**: Must use EXACT REACT class names with their specific mixed-case format
  - **Testing method**: Compared standalone evaluation with proper case (87.19%) vs lowercase (82.28%)
  - **Location**: [03_model_customization.py:331-363](experiments/03_model_customization.py#L331-L363)

- **CLIP Image Preprocessing Fix** (~6% accuracy improvement):
  - **Problem**: Explicitly setting image size distorted aspect ratios before cropping
  - **Impact**: Oxford Pets - 77.92% (distorted) → 83-87% (correct preprocessing)
  - **Root cause**: Using `size={"height": 224, "width": 224}` forced square resize, distorting images
  - **Standard CLIP**: Resize shortest edge to 224 (preserve aspect ratio), then center crop 224x224
  - **Fix**: Use processor's default parameters (no explicit size/crop)
  - **Location**: `_safe_process_images()` in [model_customizer.py:670-688](maveric/customization/model_customizer.py#L670-L688)
  - **Benefits**: Correct aspect ratio preservation, reproducible benchmarks, consistent evaluation

**November 20, 2025 - REACT-Style Text Prompting & Training Optimizations**:
- **Dataset-specific text templates**: Implemented REACT benchmark-style prompting for 15+ datasets
  - **Custom templates per dataset**: DTD, EuroSAT, FER2013, Food101, GTSRB, Oxford Flowers102, Oxford Pets, CIFAR-10/100, etc.
  - **Multiple templates**: Provides prompt diversity (e.g., "a photo of a {}", "a close-up photo of a {}")
  - **Contextually appropriate**: Each dataset gets domain-specific prompts (e.g., "satellite photo" for EuroSAT, "traffic sign" for GTSRB)
  - **Location**: `get_text_templates()` method in `elevater_datasets.py` (179 lines added)
  - **Integration**: Automatic template retrieval in `evaluation.py` and `model_customizer.py`
- **Class name normalization**: Added intelligent matching between dataset classes and template placeholders
  - **Handles format variations**: Lowercasing, underscore/hyphen to space conversion
  - **Prevents mismatches**: Ensures training data classes align with evaluation templates
  - **Location**: `model_customizer.py` (52 lines added)
- **Training hyperparameter updates**: Optimized based on empirical experiments
  - Epochs: 10 → 20 (more thorough training)
  - Learning rate: 0.0000006 → 0.0000007 (slight increase)
  - Weight decay: 0.05 → 0.07 (stronger L2 regularization)
  - Regularization weight: 0.80 → 0.75 (less MSE regularization)
  - Augmentation strength: 4 → 7 (more augmentation operations)
  - Augmentation magnitude: 15 → 22 (stronger augmentation)
  - Gradient clip value: 0.5 → 0.75 (less aggressive clipping)

**November 18, 2025 - Caltech101 Dataset Fixes**:
- **Missing "leopards" class**: Added complete missing class to Caltech101 (now correctly has 102 classes)
- **Class list formatting**: Fixed trailing comma and proper alphabetical ordering
- **Impact**: Full Caltech101 dataset support with all official classes

**November 13, 2025 - Statistics Display Improvements**:
- **Enhanced progress tracking**: Improved retrieval statistics display for better clarity
- **Consistent reporting**: Always shows cache hits and downloads (even if 0)
- **Verification formula**: Processed = Cache Hits + Downloads
- **Location**: [progress.py](maveric/core/progress.py)

**November 2, 2025 - Critical Bug Fix**:
- **Class Name Extraction Bug Fix**: Fixed critical bug in `interactive.py` where class names containing underscores (e.g., GTSRB's `ahead_only`, `beware_of_ice_snow`) were incorrectly parsed
  - **Impact**: GTSRB dataset only showed 3/43 classes (pedestrians, stop, yield) - the only classes without underscores
  - **Root Cause**: Used `split('_')[1]` which broke on class names with underscores
  - **Fix**: Proper suffix removal logic that handles arbitrary underscores in class names
  - **Affected**: Lines 275 and 1142 in `visualization/interactive.py`
  - **Tested**: All GTSRB class names now extracted correctly including complex names like `no_passing_for_vehicles_over_3_5_metric_tons`

**October 30, 2025 - Major Performance & Reliability Updates**:
- **EfficientNet Default Changed**: Now disabled by default (`enable_target_class_quality: false`) for 50-70% faster data retrieval
- **Atomic File Writes**: New `save_json_atomic()` function prevents file corruption and hanging on network filesystems (Google Drive/NFS)
- **Enhanced Cache Validation**: Automatic detection and regeneration of corrupted cache files with clear warning messages
- **Diagnostic Logging**: Comprehensive logging for file-based dataset issues, including directory structure analysis and per-class loading status
- **Progress Logging for Long Operations**: Shows progress during CLIP model loading, dataset loading, and reference generation (eliminates "frozen" appearance)
- **File-Based Dataset Bug Fix** (commit cc0b48f): Fixed "no images found" errors due to incorrect directory structure assumptions
- **Retrieval Performance Improvements** (commit 7e17c73): Multiple optimizations in retrieval module for faster processing

**Previous Improvements**:
- **Optional EfficientNet**: EfficientNet calculations can be disabled via `enable_target_class_quality: false` for ~50-70% faster retrieval (commit 8d54ac5)
- **Hierarchical file structure**: Avoids Google Drive NFS mount issues by organizing data hierarchically (commit 101170c)
- **Image copying optimization**: Pre-copies images during curation for faster validation during customization (commit 97aa1bd)
- **Enhanced progress tracking**: Improved progress bars with better timeout handling for data saving (commits 9468f77, f908a6c)
- **Debug logging**: Added detailed logs for slow validation processes (commit d961256)
- **URL tracking**: Outputs file URLs when downloads fail during curation for debugging (commit 3459d12)
- **Cleaner output**: Save data output cleaning for better console readability (commit cf27497)
- **Interactive GUI enhancements**:
  - Reset button for threshold controls
  - Random sample display on each "Show Samples" click
  - Combobox for quality threshold presets
  - EfficientNet prediction visualization tab
  - Class distribution in EfficientNet filtering

### Bug Fixes Documentation

**NEW**: Comprehensive bug fix documentation available in `docs/bugfixes/` directory:

- **`BUGFIX_SUMMARY.md`**: Complete summary of critical bug fixes implemented on October 30, 2025
  - EfficientNet default change (true → false)
  - Atomic file writes for network filesystems
  - Enhanced cache validation with corruption detection
  - Progress logging for long-running operations
  - Diagnostic logging for file-based datasets

- **`DIAGNOSTIC_LOGGING_IMPROVEMENT.md`**: Detailed guide for enhanced diagnostic logging
  - Directory structure validation
  - Per-class loading status tracking
  - Empty directory warnings
  - Comprehensive failure reports

- **`RETRIEVAL_ANALYSIS.md` series**: Complete technical analysis (23 KB total)
  - Architecture documentation with code locations
  - Performance regression analysis
  - Optimization recommendations
  - Index guide for navigating the analysis

- **`CHANGELOG_BUGFIXES.md`**: Concise changelog format for quick reference

**Performance Impact**: These bug fixes collectively provide:
- 50-70% faster data retrieval (EfficientNet disabled by default)
- Eliminated file corruption on Google Drive/NFS (atomic writes)
- Better debugging capabilities (diagnostic logging)

See [docs/bugfixes/README.md](docs/bugfixes/README.md) for detailed documentation.