# mAP Implementation Comparison: ELEVATER vs MAVERIC

## ELEVATER's Implementation

**Source**: `microsoft/vision-evaluation` library, version 0.2.9

**Key Method**: `MeanAveragePrecisionNPointsEvaluator._calc_precision_recall_interp()`

```python
def _calc_precision_recall_interp(self, predictions, targets, recall_thresholds):
    """ 
    predictions: probability/score, shape (N,)
    targets: binary ground truth {0, 1}, shape (N,)
    recall_thresholds: [1.0, 0.9, 0.8, ..., 0.1, 0.0] for 11 points
    """
    # Use sklearn to compute precision-recall curve
    precision, recall, _ = sm.precision_recall_curve(targets, predictions)
    
    # Interpolate precision at recall thresholds
    precision_interp = np.empty(len(recall_thresholds))
    recall_idx = 0
    precision_tmp = 0
    
    for idx, threshold in enumerate(recall_thresholds):
        # Find max precision where recall >= threshold
        while recall_idx < len(recall) and threshold <= recall[recall_idx]:
            precision_tmp = max(precision_tmp, precision[recall_idx])
            recall_idx += 1
        precision_interp[idx] = precision_tmp
    
    return precision_interp  # Returns 11 precision values

# Final mAP
def _calculate(self, targets, predictions, average):
    n_class = predictions.shape[1]
    recall_thresholds = np.linspace(1, 0, self.n_points, endpoint=True).tolist()
    
    # For each class, compute 11-point AP, then average
    return np.mean([
        np.mean(self._calc_precision_recall_interp(predictions[:, i], targets[:, i], recall_thresholds))
        for i in range(n_class)
    ])
```

**Recall Thresholds**: `np.linspace(1, 0, 11)` = `[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]`

## MAVERIC's Implementation

**Source**: `maveric/customization/evaluation.py::_compute_voc11_map()`

```python
def _compute_voc11_map(self, scores, labels, num_classes):
    aps = []
    
    for class_idx in range(num_classes):
        class_scores = scores[:, class_idx]
        class_labels = labels[:, class_idx]
        
        total_positives = np.sum(class_labels)
        if total_positives == 0:
            continue
        
        # Sort by score descending
        sorted_indices = np.argsort(-class_scores)
        sorted_labels = class_labels[sorted_indices]
        
        # Calculate TP and FP manually
        tp = np.cumsum(sorted_labels == 1)
        fp = np.cumsum(sorted_labels == 0)
        
        # Calculate recall and precision
        recall = tp / total_positives
        precision = tp / (tp + fp)
        
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            higher_recall_mask = recall >= t
            if np.any(higher_recall_mask):
                p = np.max(precision[higher_recall_mask])
            else:
                p = 0.0
            ap += p / 11.0
        
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0
```

**Recall Thresholds**: `np.linspace(0, 1, 11)` = `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`

## Key Differences

### 1. Precision-Recall Curve Computation

**ELEVATER**: Uses `sklearn.metrics.precision_recall_curve()`
- Automatically handles sorting by score
- Returns precision and recall arrays
- Includes special handling for edge cases

**MAVERIC**: Manual computation
- Manual sorting by score (descending)
- Manual TP/FP cumulative sum
- Manual precision/recall calculation

**Impact**: Should be equivalent, but sklearn might have edge-case handling we're missing

### 2. Recall Threshold Order

**ELEVATER**: `[1.0, 0.9, ..., 0.0]` (high to low)
**MAVERIC**: `[0.0, 0.1, ..., 1.0]` (low to high)

**Impact**: None (averaging is symmetric)

### 3. Edge Case Handling

**sklearn.precision_recall_curve** includes special handling:
- Adds sentinel values at start/end of arrays
- Handles tied scores
- Ensures recall starts at 0 and ends at 1

**MAVERIC**: Simpler approach that might miss edge cases

## Recommendation

**Use sklearn's `precision_recall_curve`** to match ELEVATER exactly:

```python
from sklearn.metrics import precision_recall_curve

def _compute_voc11_map(self, scores, labels, num_classes):
    aps = []
    recall_thresholds = np.linspace(1, 0, 11, endpoint=True)  # Match ELEVATER
    
    for class_idx in range(num_classes):
        class_scores = scores[:, class_idx]
        class_labels = labels[:, class_idx]
        
        if np.sum(class_labels) == 0:
            continue
        
        # Use sklearn (same as ELEVATER)
        precision, recall, _ = precision_recall_curve(class_labels, class_scores)
        
        # 11-point interpolation (same as ELEVATER)
        precision_interp = []
        for threshold in recall_thresholds:
            # Find max precision where recall >= threshold
            mask = recall >= threshold
            if np.any(mask):
                precision_interp.append(np.max(precision[mask]))
            else:
                precision_interp.append(0.0)
        
        aps.append(np.mean(precision_interp))
    
    return np.mean(aps) if aps else 0.0
```

This matches ELEVATER's implementation **exactly**.

## Test Priority

1. ✅ **First**: Test with softmax (no 100x scaling) using current implementation
2. ⚠️ **If still gap**: Replace with sklearn-based implementation for exact match
