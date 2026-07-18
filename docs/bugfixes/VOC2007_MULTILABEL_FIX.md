# VOC2007 Multi-Label Evaluation Fix

## Problem Summary

MAVERIC was evaluating VOC2007 as **61.36% mAP** instead of the expected **82.60% mAP** from ELEVATER benchmark.

### Root Cause (identified by Opus 4.8)

1. **Data Loading Destroyed Multi-Label Structure**: VOC2007 is inherently multi-label (one image can contain multiple objects like "dog" + "person"), but MAVERIC used `ImageFolder` which:
   - Treated each image copy in different class folders as separate samples
   - Assigned exactly one label per copy (single-label)
   - Created phantom duplicates with identical CLIP embeddings but different labels

2. **Evaluation Created One-Hot Matrix**: The evaluation code reconstructed a one-hot binary matrix from single labels instead of using true multi-hot annotations:
   ```python
   # OLD (WRONG):
   labels_binary = np.zeros((num_samples, num_classes))
   labels_binary[np.arange(num_samples), all_labels] = 1  # One positive per row
   ```

3. **Impact**: Phantom false positives at high confidence contaminated precision-recall curves, systematically underestimating mAP by ~21 percentage points.

## Solution Implemented

### 1. Created VOC2007MultiLabelDataset Class

**File**: `maveric/datasets/elevater_datasets.py` (lines 17-123)

New class that:
- Reads official VOC `ImageSets/Main/<class>_test.txt` annotation files
- Constructs proper multi-hot label vectors (shape: `[num_samples, 20]`)
- Preserves one unique image per sample (4,952 test images)
- Each image has K-hot vector where K = number of objects present

**Key Method**: `_load_annotations()`
```python
# Reads each class annotation file
for class_idx, class_name in enumerate(self.class_names):
    ann_file = imagesets_main / f'{class_name}_{self.split}.txt'
    for line in f:
        image_id, flag = parts
        if flag == 1:  # Object present
            image_labels[image_id][class_idx] = 1.0
```

### 2. Added Special VOC2007 Test Loader

**File**: `maveric/customization/model_customizer.py` (lines 386-477)

New method: `_create_voc2007_test_loader()`
- Detects VOC2007 dataset root (tries multiple common locations)
- Uses `VOC2007MultiLabelDataset` instead of `ImageFolder`
- Returns test samples with multi-hot labels
- Compatible with existing evaluation pipeline (returns 3-tuple: image, text, label)

**Integration**: Modified `_create_test_loader()` to route VOC2007 to special loader:
```python
# Special handling for VOC2007: use multi-label loader
if target_dataset_name.lower() == 'voc2007':
    return self._create_voc2007_test_loader(dataset_cache_dir, class_names)
```

### 3. Updated Evaluation to Handle Multi-Hot Labels

**File**: `maveric/customization/evaluation.py` (lines 529-550)

Changes in `voc11_map` branch:
1. **Validation**: Verify labels are multi-hot format (2D array)
   ```python
   if all_labels.ndim != 2 or all_labels.shape[1] != num_classes:
       raise ValueError("VOC 11-point mAP requires multi-label format...")
   ```

2. **Direct Use**: Pass multi-hot labels directly to `_compute_voc11_map()` (no conversion)
   ```python
   map_score = self._compute_voc11_map(all_scores, all_labels, num_classes)
   ```

3. **Skip Standard Accuracy**: Don't compute top-1 accuracy for multi-label metrics (line 561)

## Files Modified

1. **maveric/datasets/elevater_datasets.py**
   - Added `VOC2007MultiLabelDataset` class (123 lines)

2. **maveric/customization/model_customizer.py**
   - Added `_create_voc2007_test_loader()` method (92 lines)
   - Modified `_create_test_loader()` to route VOC2007 (2 lines)

3. **maveric/customization/evaluation.py**
   - Updated `voc11_map` branch to expect and validate multi-hot labels (22 lines)
   - Skip standard accuracy for multi-label metrics (1 line change)

## Expected Results

### Before Fix
```
📊 voc2007 - VOC 11-point mAP: 61.36%
```
- Used duplicated single-label samples
- One-hot label matrix (wrong)
- Contaminated precision-recall curves

### After Fix
```
✅ Using true multi-label annotations (4952 images, 20 classes)
📊 voc2007 - VOC 11-point mAP: 82.60%
```
- Uses unique images with multi-hot labels
- True VOC2007 annotations (correct)
- Matches ELEVATER benchmark

## How to Verify

### 1. Check Dataset Structure
VOC2007 dataset should be in one of these locations:
```
{cache_dir}/voc2007/datasets/VOCdevkit/VOC2007/
{cache_dir}/voc2007/datasets/VOC2007/
{cache_dir}/voc2007/VOC2007/
```

Required structure:
```
VOC2007/
├── ImageSets/
│   └── Main/
│       ├── aeroplane_test.txt
│       ├── bicycle_test.txt
│       └── ... (20 class files)
└── JPEGImages/
    └── *.jpg
```

### 2. Run Evaluation
```bash
python experiments/03_model_customization.py \
    --config maveric_config_local.yaml \
    --input /path/to/voc2007/curated/ \
    --epochs 0  # Baseline only
```

### 3. Look for Success Indicators
```
Using VOC2007MultiLabelDataset for proper multi-label evaluation
Found VOC2007 at: /content/local/cache/voc2007/datasets/VOC2007/
Loaded VOC2007 test set: 4952 images with multi-label annotations
✅ Using true multi-label annotations (4952 images, 20 classes)
📊 voc2007 - VOC 11-point mAP: ~82.60%
```

### 4. Verify Multi-Label Format
You can add debug output to check label shapes:
```python
# In evaluation.py, after line 501:
print(f"Debug: all_labels shape = {all_labels.shape}")
# Expected for VOC2007: (4952, 20) not (4952,)
```

## Backward Compatibility

- **Other datasets**: Unchanged - still use single-label format
- **VOC2007 only**: Routes to special multi-label loader
- **Config**: No changes needed - `voc11_map` metric mapping already in place
- **API**: No breaking changes to public interfaces

## Testing Checklist

- [x] VOC2007MultiLabelDataset class created
- [x] Test loader routes VOC2007 to multi-label loader
- [x] Evaluation validates multi-hot label format
- [x] Skip standard accuracy for multi-label metrics
- [x] Return 3-tuple (image, text, label) for compatibility
- [ ] Verify mAP improves from 61.36% → ~82.60%
- [ ] Check that other datasets still work correctly
- [ ] Verify no errors with missing VOC2007 data (graceful fallback)

## References

1. **ELEVATER Benchmark**: vision_benchmark/evaluation/dataset.py::Voc2007Classification
2. **PASCAL VOC 2007**: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
3. **Opus 4.8 Analysis**: Root cause identification and fix strategy
4. **VOC mAP Metric**: `_compute_voc11_map()` in evaluation.py (unchanged - was already correct)

## Notes

- The 11-point interpolated mAP implementation was already correct
- The bug was purely in data loading (destroying multi-label structure)
- This fix brings MAVERIC into compliance with ELEVATER's VOC2007 evaluation protocol
- Expected improvement: **+21.24 percentage points** (61.36% → 82.60%)
