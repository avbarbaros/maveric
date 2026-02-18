# Balance Settings Tab - User Guide

**Last Updated**: December 21, 2025

---

## Overview

The Balance Settings tab (Tab 4) in the MAVERIC interactive GUI provides tools for creating class-balanced datasets. This guide covers the recent improvements and how to use the enhanced features.

---

## Recent Updates (December 21, 2025)

### What Changed?

1. **Min Samples Widget Removed**
   - Previously: Slider to set minimum samples per class (1-100)
   - Now: Hardcoded to 1 (all classes kept, no filtering)
   - **Benefit**: Simpler UI, ensures all classes are preserved

2. **Enable Oversampling Checkbox - Full Visibility**
   - Previously: Default ipywidget width (potentially cut off)
   - Now: Explicit 500px width for full visibility
   - **Benefit**: Checkbox and label always fully visible

3. **Sorting Method Selector - NEW**
   - New dropdown: Choose how to rank samples during balancing
   - Options: **Consistency** (default) or **Weighted**
   - **Benefit**: Flexible control over sample selection quality

---

## Tab Controls

### 1. Strategy Dropdown

**Purpose**: Choose the target number of samples per class.

**Options**:
- `none`: No balancing (keep original distribution)
- `min`: Balance to smallest class size (pure undersampling)
- `max`: Balance to largest class size (pure oversampling)
- `median`: Balance to median class size (mixed)
- `mean`: Balance to mean class size (mixed)

**Example**:
```
Before balancing:
  class_a: 1,000 samples
  class_b: 5,000 samples
  class_c: 3,000 samples

Strategy: min → 1,000 samples per class
Strategy: max → 5,000 samples per class
Strategy: median → 3,000 samples per class
Strategy: mean → 3,000 samples per class
```

---

### 2. Sorting Dropdown - NEW ⭐

**Purpose**: Choose which metric to use when ranking samples during balancing.

**Options**:

#### Consistency (Default)
- Sorts samples by `consistency` score
- **Higher consistency** = better cross-modal alignment
- **Best for**: Ensuring multimodal quality (image-text agreement)

#### Weighted
- Sorts samples by `weighted_class_score`
- **Higher weighted score** = better overall quality for the class
- **Best for**: Prioritizing class-specific quality

**How It Works**:

When undersampling (reducing samples):
- Samples are sorted by the chosen metric (descending)
- Top N samples with highest scores are kept
- Lower-quality samples are discarded

When oversampling (duplicating samples):
- Samples are sorted by the chosen metric (descending)
- Best samples are duplicated first
- Maintains quality even when oversampling

**Example**:

```python
# Sample data for class_a (before balancing)
Sample 1: consistency=0.95, weighted=0.70
Sample 2: consistency=0.85, weighted=0.90
Sample 3: consistency=0.75, weighted=0.85
Sample 4: consistency=0.65, weighted=0.80

# Sorting: Consistency
# → Order: Sample 1, Sample 2, Sample 3, Sample 4

# Sorting: Weighted
# → Order: Sample 2, Sample 3, Sample 4, Sample 1

# If balancing to 2 samples per class:
# Consistency → keeps Sample 1 and Sample 2
# Weighted → keeps Sample 2 and Sample 3
```

**When to Use Each**:

| Sorting Method | Best For | Why |
|----------------|----------|-----|
| **Consistency** | General-purpose training | Ensures strong image-text alignment |
| **Weighted** | Class-specific quality | Prioritizes samples best representing each class |

---

### 3. Enable Oversampling Checkbox

**Purpose**: Allow duplication of samples when class size is below target.

**Checked (True)**:
- Classes smaller than target will be oversampled
- Best samples duplicated cyclically
- All classes reach exact target size

**Unchecked (False)**:
- Classes smaller than target keep original size
- No duplication occurs
- Result may be imbalanced

**Example**:

```
Target: 1,000 samples per class
class_a: 1,500 samples
class_b: 500 samples

Enable Oversampling = True:
  class_a: 1,000 samples (undersampled)
  class_b: 1,000 samples (oversampled - 500 duplicated)

Enable Oversampling = False:
  class_a: 1,000 samples (undersampled)
  class_b: 500 samples (kept original - no duplication)
```

---

### 4. Apply Balance Button

Click to apply the balancing strategy with current settings.

**What Happens**:
1. Reads current strategy, sorting method, and oversampling settings
2. Calculates target samples per class based on strategy
3. For each class:
   - Sorts samples by chosen metric (consistency or weighted)
   - Undersamples (keeps top N) or oversamples (duplicates best) as needed
4. Combines balanced data
5. Shuffles with fixed seed (42) for reproducibility
6. Displays before/after statistics

---

## Complete Workflow

### Step-by-Step Example

```python
from maveric.visualization import start_interactive_gui

# 1. Start GUI
gui = start_interactive_gui('cifar10')

# 2. Navigate through tabs:
#    Tab 1: Quality Thresholds → Click "Apply Settings"
#    Tab 2: Mahalanobis Filter → Optional filtering
#    Tab 3: EfficientNet Prediction → Optional filtering

# 3. Go to Tab 4: Balance Settings

# 4. Configure settings:
#    Strategy: "median"
#    Sorting: "Consistency"  (or "Weighted")
#    Enable Oversampling: ✓ (checked)

# 5. Click "Apply Balance"

# 6. View results in output
```

---

## Sorting Method Comparison

### Scenario 1: High-Quality Class-Specific Samples

**Goal**: Get samples that best represent each class

**Recommendation**: Use **Weighted** sorting
- Prioritizes `weighted_class_score`
- Selects samples with highest class-specific quality

### Scenario 2: Consistent Multimodal Samples

**Goal**: Ensure strong image-text alignment

**Recommendation**: Use **Consistency** sorting (default)
- Prioritizes `consistency` score
- Selects samples with best cross-modal agreement

### Scenario 3: General-Purpose Training

**Goal**: Balanced dataset for classifier training

**Recommendation**: Use **Consistency** sorting (default)
- More reliable for diverse tasks
- Ensures overall quality

---

## Common Use Cases

### Use Case 1: Perfect Class Balance

```
Settings:
  Strategy: min
  Sorting: Consistency
  Enable Oversampling: unchecked

Result:
  All classes have same size (smallest class)
  No artificial duplicates
  Highest-quality samples retained
```

### Use Case 2: Maximum Data Retention

```
Settings:
  Strategy: max
  Sorting: Weighted
  Enable Oversampling: checked

Result:
  All classes have same size (largest class)
  Small classes oversampled
  Class-specific quality prioritized
```

### Use Case 3: Moderate Balance

```
Settings:
  Strategy: median
  Sorting: Consistency
  Enable Oversampling: checked

Result:
  All classes have median size
  Some classes oversampled, some undersampled
  Cross-modal quality prioritized
```

---

## Technical Details

### Min Samples Behavior

**Previous Behavior**:
```python
# User could set min_samples (e.g., 50)
# Classes with < 50 samples were completely removed
min_samples = balance_min_samples_widget.value  # 1-100

if class_size < min_samples:
    # Remove entire class
    ...
```

**New Behavior**:
```python
# min_samples is now hardcoded to 1
# ALL classes are kept (no class removal)
min_samples = 1  # Hardcoded

if class_size < 1:  # Never true
    # No classes removed
    ...
```

**Impact**: All classes are preserved, even if they have very few samples.

---

### Sorting Implementation

**Code Location**: [interactive.py:501-507](maveric/visualization/interactive.py#L501-L507)

```python
# Sort by the selected sorting method for best sample selection
if sorting_method in class_data.columns:
    class_data = class_data.sort_values(sorting_method, ascending=False)
elif 'consistency' in class_data.columns:
    # Fallback to consistency if selected sorting method not available
    print(f"⚠️  Sorting method '{sorting_method}' not found, falling back to 'consistency'")
    class_data = class_data.sort_values('consistency', ascending=False)
```

**Fallback Logic**:
- If selected sorting column doesn't exist → falls back to 'consistency'
- If 'consistency' also doesn't exist → uses original order
- Warning printed if fallback occurs

---

### Oversampling Implementation

**Code Location**: [interactive.py:512-521](maveric/visualization/interactive.py#L512-L521)

```python
if current_size < target_samples:
    # Smaller than target
    if enable_oversampling:
        # Oversample by duplicating high-quality samples
        selected_data = class_data.copy()
        needed = target_samples - current_size

        # Duplicate samples cyclically (best samples duplicated first)
        for i in range(needed):
            duplicate_idx = i % current_size
            selected_data = pd.concat([selected_data, class_data.iloc[duplicate_idx:duplicate_idx+1]], ignore_index=True)
    else:
        # Keep original size
        selected_data = class_data
```

**Key Points**:
- Samples are already sorted by quality metric
- Duplicates come from top of sorted list (best quality)
- Cyclical duplication ensures even distribution

---

## Tips and Best Practices

### 1. Choose Sorting Based on Task

- **Classification training**: Use **Consistency** (ensures reliable samples)
- **Class-specific validation**: Use **Weighted** (best class representatives)
- **Not sure**: Use **Consistency** (default, safe choice)

### 2. Strategy Selection

- **Small dataset (<10K samples)**: Use `min` to avoid overfitting
- **Large dataset (>100K samples)**: Use `median` or `mean` for balance
- **Evaluation set**: Use `min` (no artificial samples)

### 3. Oversampling Decision

- **Enable** when: Training classifiers, need exact balance
- **Disable** when: Creating validation/test sets, avoiding duplicates

### 4. Monitor Class Sizes

After applying quality thresholds, check class distribution before balancing:
```python
# In Tab 1, after clicking "Apply Settings"
# Look for class distribution in output
# Helps choose appropriate balancing strategy
```

---

## Troubleshooting

### Issue 1: Sorting Method Not Working

**Symptom**: See warning "Sorting method 'weighted_class_score' not found"

**Cause**: Selected sorting column doesn't exist in data

**Solution**:
1. Check that Tab 1 (Quality Thresholds) was applied first
2. Verify `weighted_class_score` column exists in filtered_data
3. Falls back to 'consistency' automatically

---

### Issue 2: Oversampling Checkbox Not Visible

**Previous Issue**: Checkbox label was cut off

**Fixed**: Explicit 500px width now ensures full visibility

**Verify**: All three controls should have same width (500px)

---

### Issue 3: Unexpected Class Sizes

**Cause**: Oversampling disabled but expected perfect balance

**Solution**:
- Enable oversampling checkbox
- Oversampling required for classes below target

---

## Comparison with Previous Version

| Feature | Previous | Current | Benefit |
|---------|----------|---------|---------|
| Min Samples | Slider (1-100) | Hardcoded to 1 | Simpler UI, all classes kept |
| Oversampling | Checkbox (default width) | Checkbox (500px width) | Full visibility |
| Sorting | Hardcoded to consistency | Dropdown (Consistency/Weighted) | Flexible sample selection |
| Widget Count | 4 widgets | 4 widgets | Same count, better functionality |

---

## Example Output

```
⚖️  Applying Balance Strategy: median
   Sorting method: consistency
================================================================
Before balancing:
  class_a: 1,200 samples
  class_b: 3,500 samples
  class_c: 2,800 samples

Target samples per class: 2,800

After balancing:
  class_a: 2,800 samples (oversampled: +1,600)
  class_b: 2,800 samples (undersampled: -700)
  class_c: 2,800 samples (unchanged)

Total samples: 8,400
================================================================
```

---

## Summary

The updated Balance Settings tab provides:

✅ **Cleaner interface**: Min Samples removed (hardcoded to 1)
✅ **Better visibility**: Oversampling checkbox fully visible
✅ **Flexible sorting**: Choose Consistency or Weighted ranking
✅ **Same workflow**: Familiar usage pattern maintained
✅ **Better control**: Fine-tune sample selection quality

**Default Behavior**:
- Strategy: `median`
- Sorting: `Consistency`
- Oversampling: `False`
- Min Samples: `1` (all classes kept)

The new sorting dropdown gives you control over sample quality prioritization while maintaining a simple, clean interface!

---

**Last Updated**: December 21, 2025
