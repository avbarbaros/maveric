# Mahalanobis Distance Filtering - Codebase Analysis

## Overview
Complete analysis of the Mahalanobis distance filtering implementation in MAVERIC's interactive GUI, prepared for new feature development.

**Date**: January 5, 2026
**File**: `maveric/visualization/interactive.py`
**Status**: Analysis Complete - Ready for Feature Implementation

---

## Current Implementation Summary

### Key Features
✅ **Dual-Mode Filtering**: Global (all classes) and Class-Based (per-class)
✅ **Dual Input Methods**: Keep Percentile (percentage) and Keep Count (exact count)
✅ **Configurable Ideal Point**: Weighted and Consistency percentiles
✅ **Visual Analysis**: Scatter plots with Mahalanobis ellipse boundary
✅ **Class Progress Tracking**: Shows filtered classes progress (X/Y)
✅ **Reset Functionality**: Mode-aware reset for both Global and Class-Based
✅ **Grid Export**: Save filtered data as PNG grids per class

### Architecture

```
MAVERICInteractiveQualityControl
├── Data Management
│   ├── self.filtered_data (pd.DataFrame) - Current working dataset
│   ├── self.data_before_mahalanobis (pd.DataFrame) - Backup before filtering
│   └── self.class_based_filtered_data (dict) - Per-class filtered data {class_name: DataFrame}
│
├── Filter Info Storage
│   ├── self.mahalanobis_filter_info (dict) - Global filter metadata
│   └── self.mahalanobis_filter_info_class (dict) - Class-based filter metadata
│
└── Methods
    ├── _create_mahalanobis_tab() - Tab creation and UI
    ├── _apply_mahalanobis_filter() - Global filtering logic
    ├── _apply_mahalanobis_filter_class_based() - Per-class filtering logic
    ├── _plot_mahalanobis_analysis() - Global visualization
    ├── _plot_mahalanobis_analysis_class_based() - Class visualization
    ├── _show_mahalanobis_statistics() - Statistics display
    ├── _save_class_filtered_grids() - Export grids for class
    └── _consolidate_class_based_data() - Merge class data
```

---

## Data Structures

### 1. Core DataFrames

#### `self.filtered_data` (Primary Working Data)
```python
Type: pd.DataFrame
Columns:
  - label: str (class name)
  - weighted_class_score: float
  - consistency: float
  - url: str
  - text: str (caption)
  - [other quality metrics...]

Usage:
  - Primary data manipulation target
  - Gets updated by filters
  - Source for visualizations
  - Backed up before Mahalanobis filtering
```

#### `self.data_before_mahalanobis` (Backup)
```python
Type: pd.DataFrame or None
Purpose: Backup created before first Mahalanobis filter
Usage:
  - Allows reset to pre-Mahalanobis state
  - Enables repeated filtering from same baseline
  - Source for class-based filtering (prevents compounding)

Lifecycle:
  - Created on first filter application
  - Cleared on reset in Global mode
  - Persists during Class-Based filtering
```

#### `self.class_based_filtered_data` (Class Storage)
```python
Type: dict
Structure: {
    'class_name_1': DataFrame,
    'class_name_2': DataFrame,
    ...
}

Purpose: Store filtered data per class in Class-Based mode
Usage:
  - Accumulate filtered classes one by one
  - Track progress (X/Y classes filtered)
  - Source for consolidation into filtered_data
  - Cleared on reset or mode switch
```

### 2. Filter Metadata

#### `self.mahalanobis_filter_info` (Global)
```python
Type: dict
Structure: {
    'ideal_point': np.array([weighted, consistency]),
    'covariance': np.array (2x2),
    'covariance_inv': np.array (2x2),
    'threshold': float (distance threshold),
    'correlation': float (Pearson r),
    'all_samples': {
        'weighted': np.array,
        'consistency': np.array,
        'distances': np.array,
        'data_matrix': np.array
    },
    'selected_mask': np.array (boolean),
    'keep_percentile': float,
    'keep_count': int or None,
    'per_class': bool
}

Purpose: Store all data needed for visualization
Usage: _plot_mahalanobis_analysis()
```

#### `self.mahalanobis_filter_info_class` (Class-Based)
```python
Type: dict
Structure: Similar to mahalanobis_filter_info plus:
  - 'class_name': str
  - 'weighted_percentile': float
  - 'consistency_percentile': float

Purpose: Store class-specific filter metadata
Usage: _plot_mahalanobis_analysis_class_based()
```

---

## UI Components

### Tab Layout (Lines 1306-1762)

```
┌─────────────────────────────────────────────────────┐
│  Mahalanobis Distance Settings                      │
├─────────────────────────────────────────────────────┤
│  Mode: ⦿ Global  ○ Class-Based                      │
│  Class: [Dropdown - hidden in Global]               │
├─────────────────────────────────────────────────────┤
│  Weighted %ile: [95.0] (1.0-99.0)                   │
│  Consistency %ile: [95.0] (1.0-99.0)                │
│  Keep %ile: [30.0] (1.0-99.0)                       │
│  Keep Count: [1500] (integer)                       │
├─────────────────────────────────────────────────────┤
│  [Apply Filter] [Reset]                             │
│  [Add Data] [Save Filtered Data] (Class-Based only) │
├─────────────────────────────────────────────────────┤
│  Status: [HTML display]                             │
├─────────────────────────────────────────────────────┤
│  [Plot Output Area]                                 │
└─────────────────────────────────────────────────────┘
```

### Widget Definitions

#### Mode Selector (Lines 1313-1318)
```python
mode_selector = widgets.RadioButtons(
    options=['Global', 'Class-Based'],
    value='Global',
    description='Mode:',
    layout=widgets.Layout(width='300px')
)
```

#### Class Selector (Lines 1320-1331)
```python
class_selector = widgets.Dropdown(
    options=['Select class...'] + sorted_class_names,
    value='Select class...',
    description='Class:',
    layout=widgets.Layout(width='250px', visibility='hidden'),
    style={'description_width': '50px'}
)
# Visibility toggled by mode_selector
```

#### Percentile Inputs (Lines 1334-1364)
```python
# Ideal point configuration
weighted_percentile_text = FloatText(value=95.0, min=1.0, max=99.0)
consistency_percentile_text = FloatText(value=95.0, min=1.0, max=99.0)

# Filter amount
keep_percentile_text = FloatText(value=30.0, min=1.0, max=99.0)
keep_count_text = IntText(value=0, min=1)
# Auto-sync: percentile ↔ count
```

#### Buttons (Lines 1422-1448)
```python
apply_button = Button(description='Apply Filter', button_style='primary')
add_data_button = Button(description='Add Data', button_style='success')  # Class-Based only
save_filtered_button = Button(description='Save Filtered Data', button_style='warning')  # Class-Based only
reset_button = Button(description='Reset', button_style='danger')
```

---

## Key Methods

### 1. Filtering Logic

#### `_apply_mahalanobis_filter()` (Lines 1764-1906)
**Purpose**: Apply global or per-class Mahalanobis filtering

**Signature**:
```python
def _apply_mahalanobis_filter(self,
                              keep_percentile=None,
                              keep_count=None,
                              weighted_percentile=95,
                              consistency_percentile=95,
                              per_class=False)
```

**Algorithm**:
1. Validate data and required columns
2. Create backup (`data_before_mahalanobis`)
3. Extract `weighted_class_score` and `consistency` metrics
4. Calculate ideal point using percentiles
5. Compute covariance matrix and inverse
6. Calculate Mahalanobis distances for all samples
7. Determine threshold based on `keep_count` or `keep_percentile`
8. Apply mask and update `filtered_data`
9. Store metadata in `mahalanobis_filter_info`

**Key Code Snippet**:
```python
# Priority logic: count > percentile
if keep_count is not None and keep_count > 0:
    n_keep = min(keep_count, len(df))
else:
    n_keep = max(1, int(len(df) * keep_percentile / 100))

threshold = np.partition(distances, n_keep-1)[n_keep-1]
mask = distances <= threshold
filtered_df = df[mask].copy()
```

#### `_apply_mahalanobis_filter_class_based()` (Lines 2033-2162)
**Purpose**: Apply Mahalanobis filtering to specific class only

**Key Differences from Global**:
- Filters from `data_before_mahalanobis` (not `filtered_data`)
- Stores result in `class_based_filtered_data[class_name]`
- Creates class-specific metadata (`mahalanobis_filter_info_class`)
- Does NOT update `filtered_data` directly (only on consolidation)

**Usage Flow**:
```
1. User selects class → Apply Filter
2. Filtered data stored in class_based_filtered_data[class]
3. User clicks "Add Data"
4. _consolidate_class_based_data() merges all classes → filtered_data
```

### 2. Visualization

#### `_plot_mahalanobis_analysis()` (Lines 1908-2013)
**Purpose**: Create comprehensive visualization for global filter

**Plot Components**:
```
┌────────────────┬───────────────────────┐
│  Top Histogram │                       │
│  (Weighted)    │                       │
├────────────────┼───────────────────────┤
│                │                       │
│                │   Main Scatter        │
│                │   - Gray dots (reject)│
│                │   - Green dots (keep) │
│                │   - Red star (ideal)  │
│  Right Hist    │   - Red ellipse       │
│  (Consistency) │   - ρ correlation     │
│                │                       │
└────────────────┴───────────────────────┘
```

**Created using**:
- `matplotlib.pyplot` with GridSpec layout
- `matplotlib.patches.Ellipse` for boundary
- Scatter plots with alpha transparency
- Histograms with density normalization

#### `_plot_mahalanobis_analysis_class_based()` (Lines 2163-2315)
**Purpose**: Same as global but uses `mahalanobis_filter_info_class`

---

## Callbacks and Event Handling

### 1. Mode Change (Lines 1459-1481)
```python
def on_mode_change(change):
    if change['new'] == 'Class-Based':
        # Show: class_selector, add_data_button, save_filtered_button
        class_selector.layout.visibility = 'visible'
        add_data_button.layout.visibility = 'visible'
        save_filtered_button.layout.visibility = 'visible'
    else:  # Global
        # Hide class-specific controls
        class_selector.layout.visibility = 'hidden'
        ...

    # Update keep_count for new mode
    update_keep_count()
```

### 2. Apply Filter (Lines 1494-1596)
**Global Mode Flow**:
```
1. Reset to data_before_mahalanobis (if exists)
2. Call _apply_mahalanobis_filter()
3. Plot results with _plot_mahalanobis_analysis()
4. Show statistics
5. Update status display
```

**Class-Based Flow**:
```
1. Validate class selection
2. Call _apply_mahalanobis_filter_class_based()
3. Plot with _plot_mahalanobis_analysis_class_based()
4. Update status (samples kept for this class)
5. Data stored in class_based_filtered_data (NOT filtered_data yet)
```

### 3. Add Data (Lines 1598-1644) - Class-Based Only
```python
def on_add_data_clicked(b):
    # Consolidate all filtered classes into filtered_data
    _consolidate_class_based_data()

    # Show progress: X/Y classes filtered
    print(f"📋 Filtered classes ({num_filtered}/{total_classes}): {class_names}")
    print(f"⏳ Remaining classes to filter: {remaining}")
```

### 4. Save Filtered Data (Lines 1646-1686) - Class-Based Only
```python
def on_save_filtered_clicked(b):
    # Export grid PNGs: {dataset}_{class}_{seq}.png
    _save_class_filtered_grids(selected_class)
```

### 5. Reset (Lines 1688-1747)
**Mode-Aware Reset**:
```python
if mode == 'Global':
    # Restore from backup
    filtered_data = data_before_mahalanobis.copy()
    data_before_mahalanobis = None
else:  # Class-Based
    if no_class_selected:
        # Clear all class data
        class_based_filtered_data.clear()
    else:
        # Remove specific class
        del class_based_filtered_data[class_name]
        _consolidate_class_based_data()  # Re-merge remaining
```

### 6. Percentile ↔ Count Sync (Lines 1398-1413)
```python
# Auto-update count when percentile changes
def on_percentile_change(change):
    total_samples = get_sample_count_for_filter()
    count = int(total_samples * change['new'] / 100.0)
    keep_count_text.value = max(1, count)

# Auto-update percentile when count changes
def on_count_change(change):
    total_samples = get_sample_count_for_filter()
    percentile = (change['new'] / total_samples) * 100.0
    keep_percentile_text.value = round(percentile, 1)
```

---

## Data Flow Diagrams

### Global Mode Flow
```
┌─────────────────┐
│ filtered_data   │ (from Tab 1)
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│ data_before_mahalanobis │ ← Backup created
└────────┬────────────────┘
         │
         ↓
   [Apply Filter]
         │
         ↓
┌─────────────────────────────────────┐
│ _apply_mahalanobis_filter()         │
│  - Calculate distances              │
│  - Apply threshold (count/percentile)│
│  - Create mask                       │
└────────┬────────────────────────────┘
         │
         ↓
┌─────────────────┐
│ filtered_data   │ ← Updated with filtered samples
└────────┬────────┘
         │
         ↓
  [Visualization]
```

### Class-Based Mode Flow
```
┌─────────────────┐
│ filtered_data   │ (from Tab 1)
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│ data_before_mahalanobis │ ← Backup (once)
└────────┬────────────────┘
         │
         ├──────────────────┐
         │                  │
    [Class A]          [Class B]
         │                  │
         ↓                  ↓
  Filter Class A      Filter Class B
         │                  │
         ↓                  ↓
┌──────────────────────────────────┐
│ class_based_filtered_data        │
│  {                               │
│    'airplane': DataFrame (350),  │
│    'automobile': DataFrame (400),│
│    ...                           │
│  }                               │
└────────┬─────────────────────────┘
         │
         ↓
  [Add Data] × N classes
         │
         ↓
┌──────────────────────────┐
│ _consolidate_class_based │
│  pd.concat(all_classes)  │
└────────┬─────────────────┘
         │
         ↓
┌─────────────────┐
│ filtered_data   │ ← Final merged data
└─────────────────┘
```

---

## Critical Implementation Details

### 1. Backup Creation (Lines 1794-1795, 2052-2054)
```python
# Global mode
if self.data_before_mahalanobis is None or per_class:
    self.data_before_mahalanobis = self.filtered_data.copy()

# Class-Based mode (always from backup)
if self.data_before_mahalanobis is None:
    self.data_before_mahalanobis = self.filtered_data.copy()
```

**Why Important**: Prevents compounding filters in class-based mode. Each class filters from the SAME baseline.

### 2. Count vs Percentile Priority (Lines 1870-1876, 2122-2127)
```python
if keep_count is not None and keep_count > 0:
    n_keep = min(keep_count, len(df))  # Use count
    print(f"🎯 Keeping exactly {n_keep:,} samples")
else:
    n_keep = max(1, int(len(df) * keep_percentile / 100))  # Use percentile
    print(f"🎯 Keeping top {keep_percentile}%")
```

**Priority**: `keep_count > 0` → use count, else → use percentile

### 3. Threshold Calculation (Lines 1877, 2129)
```python
threshold = np.partition(distances, n_keep-1)[n_keep-1]
mask = distances <= threshold
```

**np.partition()**: O(n) average, finds k-th smallest element without full sort.

### 4. Class Consolidation (Lines 2285-2321)
```python
def _consolidate_class_based_data(self):
    """Merge all filtered classes into filtered_data"""
    all_data = []
    for class_name in sorted(self.class_based_filtered_data.keys()):
        all_data.append(self.class_based_filtered_data[class_name])

    self.filtered_data = pd.concat(all_data, ignore_index=True)
    print(f"✅ Consolidated {len(all_data)} classes → {len(self.filtered_data):,} samples")
```

**Called**: After "Add Data" button click

---

## Configuration Points

### Default Values
```python
weighted_percentile_text.value = 95.0    # Ideal point weighted axis
consistency_percentile_text.value = 95.0  # Ideal point consistency axis
keep_percentile_text.value = 30.0        # Keep top 30%
keep_count_text.value = 0                # 0 = use percentile
mode_selector.value = 'Global'           # Default mode
```

### Validation Ranges
```python
weighted_percentile: 1.0 - 99.0 (float)
consistency_percentile: 1.0 - 99.0 (float)
keep_percentile: 1.0 - 99.0 (float)
keep_count: min=1 (integer)
```

---

## Extension Points for New Features

### 1. Additional Filter Parameters
**Location**: Lines 1334-1374 (widget definitions)

**Pattern**:
```python
# Add new control
new_param_widget = widgets.FloatText(
    value=default_value,
    description='Param Name:',
    layout=widgets.Layout(width='200px'),
    style={'description_width': '100px'}
)

# Update filter methods signatures
def _apply_mahalanobis_filter(..., new_param=default):
    # Use new_param in calculation
    ...
```

### 2. Alternative Distance Metrics
**Location**: Lines 1827-1830 (distance calculation)

**Current**: Mahalanobis distance
```python
distances = np.array([
    mahalanobis(x, ideal_point, covariance_inv)
    for x in data_matrix
])
```

**Extension**: Add metric selector
```python
if metric_type == 'mahalanobis':
    distances = mahalanobis_distances(...)
elif metric_type == 'euclidean':
    distances = euclidean_distances(...)
elif metric_type == 'cosine':
    distances = cosine_distances(...)
```

### 3. Multi-Metric Filtering
**Location**: Lines 1797-1800 (data extraction)

**Current**: Uses only `weighted_class_score` and `consistency`
```python
weighted = df['weighted_class_score'].values
consistency = df['consistency'].values
```

**Extension**: Add metric selector
```python
available_metrics = ['weighted_class_score', 'consistency',
                     'sharpness_score', 'resolution_score']
selected_metrics = metric_selector.value  # Multi-select widget
data_matrix = df[selected_metrics].values
```

### 4. Advanced Visualization Options
**Location**: Lines 1908-2013 (plotting)

**Possible Additions**:
- Contour plot overlay
- 3D scatter plot (3 metrics)
- Interactive plotly visualization
- Export plot to file
- Density heatmap

### 5. Batch Class Processing
**New Feature Idea**: Auto-filter all classes with same parameters

**Implementation**:
```python
batch_process_button = Button(description='Batch Process All Classes')

def on_batch_process(b):
    for class_name in all_classes:
        result = _apply_mahalanobis_filter_class_based(
            class_name, keep_percentile, keep_count, ...
        )
        if result:
            class_based_filtered_data[class_name] = result
    _consolidate_class_based_data()
```

---

## Testing Strategy

### Unit Tests Needed
1. **Backup Creation**: Verify `data_before_mahalanobis` created correctly
2. **Count Priority**: Test count > 0 uses exact count
3. **Percentile Fallback**: Test count = 0 uses percentile
4. **Threshold Calculation**: Verify exactly N samples kept
5. **Class Consolidation**: Test merge preserves all data
6. **Reset Logic**: Test mode-aware reset behavior

### Integration Tests Needed
1. **Global → Class-Based Switch**: Verify data integrity
2. **Multiple Class Filtering**: Test sequential class filtering
3. **Reset After Filtering**: Verify restore to baseline
4. **Percentile ↔ Count Sync**: Test bidirectional updates
5. **Empty Data Handling**: Test error handling

### Manual Testing Checklist
- [ ] Load dataset (CIFAR-10)
- [ ] Global mode: Apply filter with percentile
- [ ] Global mode: Apply filter with count
- [ ] Switch to Class-Based mode
- [ ] Filter 3 classes sequentially
- [ ] Add Data after each class
- [ ] Check progress tracking
- [ ] Save Filtered Data
- [ ] Reset in Class-Based (specific class)
- [ ] Reset in Class-Based (all data)
- [ ] Switch back to Global
- [ ] Reset in Global mode

---

## Performance Considerations

### Current Bottlenecks
1. **Distance Calculation**: O(n) per sample, can be slow for large datasets
2. **Visualization**: Creating scatter plots with 50K+ points
3. **Grid Export**: Loading and processing images for PNG grids

### Optimization Opportunities
1. **Vectorized Distance**: Use scipy.spatial.distance.mahalanobis with broadcasting
2. **Downsampling Plots**: Show subset of points for large datasets
3. **Async Grid Export**: Generate grids in background thread
4. **Caching**: Cache distance calculations per parameter set

### Memory Usage
- **filtered_data**: O(n × m) where n=samples, m=columns
- **data_before_mahalanobis**: O(n × m) backup copy
- **class_based_filtered_data**: O(k × n/k × m) ≈ O(n × m) total
- **Distance arrays**: O(n) per filter operation

**Estimate**: 50K samples × 30 columns × 8 bytes ≈ 12 MB per DataFrame

---

## Dependencies

### External Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import mahalanobis
from ipywidgets import widgets
from IPython.display import display, clear_output
```

### Internal Imports
```python
from maveric.visualization.interactive import MAVERICInteractiveQualityControl
```

---

## Documentation References

- **Keep Count Feature**: [KEEP_COUNT_IMPLEMENTATION.md](KEEP_COUNT_IMPLEMENTATION.md)
- **Class-Based Mode**: [MAHALANOBIS_CLASS_BASED_MODE.md](MAHALANOBIS_CLASS_BASED_MODE.md)
- **Reset Button**: [MAHALANOBIS_RESET_BUTTON.md](MAHALANOBIS_RESET_BUTTON.md)
- **CLAUDE.md Entry**: Lines 7-82 (Recent updates section)

---

## Summary

### Current State
✅ **Fully Functional**: Global and Class-Based filtering working
✅ **Well Documented**: Comprehensive docs for all features
✅ **User-Friendly**: Dual input methods, progress tracking, reset
✅ **Robust**: Error handling, validation, backup system

### Code Quality
- **Lines of Code**: ~500 lines for Mahalanobis tab
- **Complexity**: Moderate (multiple modes, callbacks, state management)
- **Maintainability**: Good (clear method separation, documented)
- **Test Coverage**: Manual testing documented, unit tests recommended

### Ready for Extension
The codebase is well-structured for adding new features:
- Clear separation of concerns (UI, logic, visualization)
- Consistent naming conventions
- Documented extension points
- Modular callback system

---

**Analysis Complete**: Ready for feature implementation discussions!
**Contact**: Review this analysis before implementing new features
**Updated**: January 5, 2026
