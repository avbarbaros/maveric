# Keep Count/Percentile Dual Input Feature

**Date**: December 22, 2025  
**Feature**: Dual input for Keep Percentile and Keep Count  
**Status**: ✅ Implemented

---

## Overview

Added a "Keep Count" text box alongside the existing "Keep Percentile" box in the Mahalanobis Filter tab. Both inputs are synchronized - changing one automatically updates the other based on the total number of samples.

---

## Features

### Two Input Methods

**1. Keep Percentile** (existing):
- Enter percentage (1-99%)
- Example: 30% keeps top 30% of samples
- Good for: Relative filtering across different datasets

**2. Keep Count** (NEW):
- Enter absolute number of samples
- Example: 500 keeps exactly 500 samples
- Good for: Precise sample counts, fixed dataset sizes

### Automatic Synchronization

When you change one input, the other updates automatically:

**Change Percentile → Updates Count**:
```
Total samples: 1000
Keep %ile: 30.0 → Keep Count: 300 (auto-calculated)
Keep %ile: 45.0 → Keep Count: 450 (auto-calculated)
```

**Change Count → Updates Percentile**:
```
Total samples: 1000
Keep Count: 500 → Keep %ile: 50.0 (auto-calculated)
Keep Count: 250 → Keep %ile: 25.0 (auto-calculated)
```

---

## UI Layout

### Before (Single Input)
```
┌────────────────────────────────────────────────────┐
│ Weighted %ile: [95.0]  Consistency %ile: [95.0]   │
│ Keep %ile: [30.0]                                  │
└────────────────────────────────────────────────────┘
```

### After (Dual Input)
```
┌────────────────────────────────────────────────────┐
│ Weighted %ile: [95.0]  Consistency %ile: [95.0]   │
│ Keep %ile: [30.0]      Keep Count: [150]          │
│    ↕ Auto-synchronized ↕                           │
└────────────────────────────────────────────────────┘
```

---

## How It Works

### Calculation Logic

**Percentile → Count**:
```python
count = int(total_samples * percentile / 100.0)
count = max(1, count)  # Minimum 1 sample
```

**Count → Percentile**:
```python
percentile = (count / total_samples) * 100.0
percentile = min(99.0, max(1.0, percentile))  # Clamp to 1-99%
percentile = round(percentile, 1)  # Round to 1 decimal
```

### Sample Count Detection

The system automatically detects how many samples will be filtered:

**Global Mode**:
```python
source_data = data_before_mahalanobis or filtered_data
total_samples = len(source_data)
```

**Class-Based Mode**:
```python
source_data = data_before_mahalanobis or filtered_data
class_data = source_data[source_data['label'] == selected_class]
total_samples = len(class_data)
```

---

## Usage Examples

### Example 1: Global Mode (50,000 samples)

**Using Percentile**:
```
Keep %ile: 30.0 → Keep Count: 15,000 (auto)
Keep %ile: 20.0 → Keep Count: 10,000 (auto)
Keep %ile: 40.0 → Keep Count: 20,000 (auto)
```

**Using Count**:
```
Keep Count: 5,000 → Keep %ile: 10.0 (auto)
Keep Count: 15,000 → Keep %ile: 30.0 (auto)
Keep Count: 25,000 → Keep %ile: 50.0 (auto)
```

---

### Example 2: Class-Based Mode - "airplane" (500 samples)

**Using Percentile**:
```
Keep %ile: 40.0 → Keep Count: 200 (auto)
Keep %ile: 30.0 → Keep Count: 150 (auto)
Keep %ile: 50.0 → Keep Count: 250 (auto)
```

**Using Count**:
```
Keep Count: 200 → Keep %ile: 40.0 (auto)
Keep Count: 100 → Keep %ile: 20.0 (auto)
Keep Count: 300 → Keep %ile: 60.0 (auto)
```

---

### Example 3: Class-Based Mode - "automobile" (500 samples)

**Switch from airplane to automobile**:
```
Mode: Class-Based, Class: airplane (500 samples)
  Keep %ile: 40.0 → Keep Count: 200

Switch to: automobile (500 samples)
  Keep %ile: 40.0 → Keep Count: 200 (recalculated)
  (Same percentile, same count because same class size)

If automobile had 600 samples:
  Keep %ile: 40.0 → Keep Count: 240 (recalculated!)
```

---

## Dynamic Updates

### When Count Updates Automatically

1. **On Tab Load**: Initializes based on default 30% percentile
2. **Mode Change**: Global ↔ Class-Based
3. **Class Change**: Different class selected in dropdown
4. **Percentile Change**: User modifies Keep %ile

### When Percentile Updates Automatically

1. **Count Change**: User modifies Keep Count

---

## Implementation Details

**Location**: [interactive.py:1366-1491](maveric/visualization/interactive.py#L1366-L1491)

### Widgets Created

```python
# Keep percentile (existing)
keep_percentile_text = widgets.FloatText(
    value=30.0,
    min=1.0,
    max=99.0,
    step=0.1,
    description='Keep %ile:',
    layout=widgets.Layout(width='200px')
)

# Keep count (NEW)
keep_count_text = widgets.IntText(
    value=0,
    min=1,
    description='Keep Count:',
    layout=widgets.Layout(width='200px')
)
```

### Helper Function

```python
def get_sample_count_for_filter():
    """Get the number of samples that will be filtered"""
    if self.filtered_data is None or len(self.filtered_data) == 0:
        return 0

    mode = mode_selector.value
    if mode == 'Class-Based':
        selected_class = class_selector.value
        if selected_class == 'Select class...':
            return 0

        source_data = self.data_before_mahalanobis if self.data_before_mahalanobis is not None else self.filtered_data
        class_data = source_data[source_data['label'] == selected_class]
        return len(class_data)
    else:
        source_data = self.data_before_mahalanobis if self.data_before_mahalanobis is not None else self.filtered_data
        return len(source_data)
```

### Observers

```python
# Percentile → Count
def on_percentile_change(change):
    total_samples = get_sample_count_for_filter()
    if total_samples > 0:
        count = int(total_samples * change['new'] / 100.0)
        keep_count_text.value = max(1, count)

# Count → Percentile
def on_count_change(change):
    total_samples = get_sample_count_for_filter()
    if total_samples > 0 and change['new'] > 0:
        percentile = (change['new'] / total_samples) * 100.0
        percentile = min(99.0, max(1.0, percentile))
        keep_percentile_text.value = round(percentile, 1)

keep_percentile_text.observe(on_percentile_change, names='value')
keep_count_text.observe(on_count_change, names='value')
```

---

## Benefits

✅ **Flexibility**: Choose percentile OR absolute count  
✅ **Precision**: Exact sample counts when needed  
✅ **Auto-sync**: No manual calculation required  
✅ **Context-aware**: Updates based on mode/class  
✅ **User-friendly**: Use whichever input makes sense  
✅ **Real-time**: Instant feedback on changes  

---

## Use Cases

### Use Case 1: Fixed Dataset Size
**Scenario**: Need exactly 1,000 samples for training
**Solution**: Set Keep Count = 1,000
**Result**: Percentile auto-adjusts (e.g., 20% for 5K samples)

### Use Case 2: Relative Filtering
**Scenario**: Keep top 30% across all classes
**Solution**: Set Keep %ile = 30.0
**Result**: Count auto-adjusts per class size

### Use Case 3: Budget-Constrained
**Scenario**: Can only annotate 500 samples
**Solution**: Set Keep Count = 500
**Result**: Gets you closest to budget

### Use Case 4: Compare Approaches
**Scenario**: Try 20%, 30%, 40% to see quality
**Solution**: Adjust Keep %ile, see exact counts
**Result**: Easy A/B testing

---

## Edge Cases

### Very Small Counts
```
Total samples: 10
Keep Count: 1 → Keep %ile: 10.0
Keep Count: 5 → Keep %ile: 50.0
Keep Count: 9 → Keep %ile: 90.0
```

### Very Large Counts
```
Total samples: 100,000
Keep Count: 50,000 → Keep %ile: 50.0
Keep Count: 75,000 → Keep %ile: 75.0
Keep Count: 99,000 → Keep %ile: 99.0
```

### Rounding
```
Total samples: 1,000
Keep %ile: 33.3 → Keep Count: 333
Keep Count: 333 → Keep %ile: 33.3
```

### Class Size Variations
```
Class A (1,000 samples): 30% = 300 samples
Class B (500 samples):   30% = 150 samples
Class C (2,000 samples): 30% = 600 samples
```
(Same percentile, different counts - auto-adjusted)

---

## Summary

The dual Keep Percentile/Keep Count input provides flexible filtering options. Users can think in terms of relative percentages (30% of samples) or absolute counts (500 samples), whichever is more natural for their workflow. The automatic synchronization ensures both values stay consistent, and the context-aware updates handle mode and class changes seamlessly. 🎯

---

**Files Modified**: 1  
**Lines Changed**: ~90  
**Test Status**: Ready for testing  
**User Requested**: Yes ✅
