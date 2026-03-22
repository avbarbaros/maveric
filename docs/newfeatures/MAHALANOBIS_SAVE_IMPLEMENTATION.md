# Mahalanobis Filter Save Plot Implementation

## Overview

Added "Save Plot" functionality to the Mahalanobis Filter tab in the interactive data curation GUI. Users can now save plots in multiple formats (EPS, PNG, PDF, SVG) along with a CSV file containing all data points.

## Implementation Summary

### Changes Made

#### 1. New Helper Method: `_create_mahalanobis_figure()`
**Location**: `maveric/visualization/interactive.py` (after line 2015)

**Purpose**: Creates Mahalanobis plot figure without displaying it, enabling both display and save functionality.

**Features**:
- Supports both Global and Class-Based modes
- Returns matplotlib Figure object (or None if no data)
- Consolidates plotting code from both `_plot_mahalanobis_analysis()` and `_plot_mahalanobis_analysis_class_based()`

**Parameters**:
- `mode`: 'Global' or 'Class-Based'
- `class_name`: Required for Class-Based mode

**Returns**: matplotlib Figure object

#### 2. Refactored Plotting Methods

**`_plot_mahalanobis_analysis()`**:
- Now calls `_create_mahalanobis_figure(mode='Global')`
- Displays figure with `plt.show()`
- Simplified from ~85 lines to ~10 lines

**`_plot_mahalanobis_analysis_class_based()`**:
- Now calls `_create_mahalanobis_figure(mode='Class-Based', class_name=...)`
- Displays figure with `plt.show()`
- Simplified from ~70 lines to ~10 lines

#### 3. New Save Method: `_save_mahalanobis_plot_and_data()`
**Location**: After `_consolidate_class_based_data()` method

**Purpose**: Saves both plot file and CSV data file.

**Features**:
- **Plot saving**:
  - Supports EPS, PNG, PDF, SVG formats
  - High quality: 300 DPI, tight bounding box
  - Automatic directory creation
  - Safe filename generation (sanitizes special characters)
- **CSV saving**:
  - Exports all data points with columns:
    - `weighted_class_score`: X-axis values
    - `consistency`: Y-axis values
    - `mahalanobis_distance`: Distance from ideal point
    - `selected`: Boolean (True = selected, False = rejected)
  - Sorted by selection status, then by distance
  - 6 decimal precision for float values

**Parameters**:
- `mode`: 'Global' or 'Class-Based'
- `class_name`: Required for Class-Based mode
- `file_format`: 'eps', 'png', 'pdf', or 'svg'

**Returns**: `(plot_path, csv_path)` tuple or `(None, None)` on error

**Output Files**:
```
Global mode:
  - {dataset}_mahalanobis_global.{ext}
  - {dataset}_mahalanobis_global_data.csv

Class-Based mode:
  - {dataset}_mahalanobis_{class}.{ext}
  - {dataset}_mahalanobis_{class}_data.csv

Example:
  - cifar10_mahalanobis_global.eps
  - cifar10_mahalanobis_global_data.csv
  - cifar10_mahalanobis_airplane.eps
  - cifar10_mahalanobis_airplane_data.csv
```

#### 4. New GUI Components

**Format Selector Dropdown**:
```python
format_selector = widgets.Dropdown(
    options=['EPS', 'PNG', 'PDF', 'SVG'],
    value='EPS',
    description='Format:',
    layout=widgets.Layout(width='150px')
)
```

**Save Plot Button**:
```python
save_plot_button = widgets.Button(
    description='Save Plot',
    button_style='info',
    icon='download',
    layout=widgets.Layout(width='120px')
)
```

**Button Callback**: `on_save_plot_clicked()`
- Validates filter info exists
- Gets mode and format from GUI
- Calls `_save_mahalanobis_plot_and_data()`
- Updates status display with success/error message

#### 5. Updated GUI Layout

**New Row Added**:
```python
widgets.HBox([
    format_selector,
    save_plot_button
], layout=widgets.Layout(margin='5px 0'))
```

**Complete Layout Order**:
1. Mode selector + Class selector
2. Weighted percentile + Consistency percentile
3. Keep percentile + Keep count
4. Apply + Add Data + Save Filtered Data + Reset buttons
5. **Format selector + Save Plot button** ← NEW
6. Status display
7. Plot output

## Usage Guide

### Global Mode

1. **Apply Filter**:
   - Select "Global" mode
   - Set percentiles (Weighted: 95, Consistency: 95, Keep: 30)
   - Click "Apply Filter"
   - Plot appears below

2. **Save Plot and Data**:
   - Select format from dropdown (EPS, PNG, PDF, SVG)
   - Click "Save Plot"
   - Console shows:
     ```
     ✅ Plot saved: ./results/cifar10/curationResults/cifar10_mahalanobis_global.eps
     ✅ Data saved: ./results/cifar10/curationResults/cifar10_mahalanobis_global_data.csv
        Total points: 50,000
        Selected: 15,000
        Rejected: 35,000
     ```
   - Status display shows: "✅ Plot and data saved | Format: EPS"

### Class-Based Mode

1. **Apply Filter**:
   - Select "Class-Based" mode
   - Select class from dropdown (e.g., "airplane")
   - Set percentiles
   - Click "Apply Filter"
   - Class-specific plot appears

2. **Save Plot and Data**:
   - Select format (default: EPS)
   - Click "Save Plot"
   - Console shows:
     ```
     ✅ Plot saved: ./results/cifar10/curationResults/cifar10_mahalanobis_airplane.eps
     ✅ Data saved: ./results/cifar10/curationResults/cifar10_mahalanobis_airplane_data.csv
        Total points: 5,000
        Selected: 1,500
        Rejected: 3,500
     ```
   - Status display shows: "✅ Plot and data saved for 'airplane' | Format: EPS"

## CSV File Format

### Structure

```csv
weighted_class_score,consistency,mahalanobis_distance,selected
0.850000,0.920000,1.234567,True
0.830000,0.910000,1.456789,True
0.820000,0.900000,1.678901,True
...
0.450000,0.720000,8.901234,False
0.440000,0.710000,9.123456,False
0.430000,0.700000,9.345678,False
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `weighted_class_score` | float | X-axis value (weighted similarity score) |
| `consistency` | float | Y-axis value (consistency metric) |
| `mahalanobis_distance` | float | Distance from ideal point |
| `selected` | boolean | True if sample is selected, False if rejected |

### Sorting

- **Primary**: By `selected` (selected samples first)
- **Secondary**: By `mahalanobis_distance` (ascending)

**Result**: Selected samples sorted by quality (best first), followed by rejected samples sorted by distance (closest first).

## File Output Structure

```
{base_dir}/
├── raw/                          # Original retrieval results
│   └── ...
├── images/                       # Cached images
│   └── ...
└── curationResults/              # All curation outputs
    ├── cifar10_grid_001.png                        # Existing: sample grids
    ├── cifar10_airplane_001.png                    # Existing: class grids
    ├── cifar10_mahalanobis_global.eps              # NEW: Global plot (EPS)
    ├── cifar10_mahalanobis_global_data.csv         # NEW: Global data (CSV)
    ├── cifar10_mahalanobis_airplane.eps            # NEW: Class plot (EPS)
    ├── cifar10_mahalanobis_airplane_data.csv       # NEW: Class data (CSV)
    ├── cifar10_mahalanobis_global.png              # NEW: Global plot (PNG)
    ├── cifar10_mahalanobis_airplane.pdf            # NEW: Class plot (PDF)
    └── training_rotation_001.json                  # Existing: training data
```

## Format Comparison

| Format | Type | Size (typical) | Use Case |
|--------|------|----------------|----------|
| **EPS** | Vector | 50-500 KB | Publication-ready, LaTeX documents, scientific papers |
| **PDF** | Vector | 50-500 KB | Modern documents, presentations, sharing |
| **SVG** | Vector | 50-500 KB | Web use, editing in Inkscape/Illustrator |
| **PNG** | Raster | 200-800 KB | Quick preview, web use, presentations |

**Recommendation**: Use **EPS** for academic publications, **PDF** for general sharing, **PNG** for quick previews.

## Error Handling

### Common Errors

**Error**: "❌ No Mahalanobis filter info available. Apply filter first."
- **Cause**: Clicked "Save Plot" before clicking "Apply Filter"
- **Solution**: Click "Apply Filter" to generate plot first

**Error**: "❌ Select a class before saving plot."
- **Cause**: In Class-Based mode, no class selected
- **Solution**: Select a class from dropdown

**Error**: "❌ Failed to create Mahalanobis plot"
- **Cause**: Filter info exists but figure creation failed
- **Solution**: Check console for detailed error, may need to re-apply filter

### File Permission Issues

If save fails due to permissions:
1. Check `curationResults/` directory is writable
2. Verify disk space available
3. On Google Drive/NFS, ensure mount is active

## Technical Details

### Plot Quality Settings

```python
fig.savefig(filepath,
           format=file_format,
           bbox_inches='tight',  # Trim whitespace
           dpi=300)              # High resolution (publication quality)
```

### Filename Sanitization

Special characters replaced with underscores:
- `/` → `_`
- `\` → `_`
- Space → `_`

**Example**: Class name `"aquarium fish"` → filename `cifar10_mahalanobis_aquarium_fish.eps`

### CSV Float Precision

All float values saved with 6 decimal places:
```python
data_df.to_csv(csv_path, index=False, float_format='%.6f')
```

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Helper method | `interactive.py` | 2015-2106 |
| Save method | `interactive.py` | 2447-2531 |
| Format selector | `interactive.py` | 1449-1456 |
| Save button | `interactive.py` | 1458-1463 |
| Save callback | `interactive.py` | 1859-1914 |
| Layout update | `interactive.py` | 1922-1945 |

## Testing Checklist

### Functional Tests

- [x] Global mode: Apply filter → Save EPS → Verify files created
- [x] Global mode: Save PNG, PDF, SVG → Verify correct formats
- [x] Class-Based mode: Apply filter for one class → Save → Verify files
- [x] Class-Based mode: Save multiple classes → Verify unique filenames
- [x] CSV file: Open in spreadsheet → Verify 4 columns, correct sorting
- [x] Error handling: Click save without applying filter → Error message shown
- [x] Error handling: Class-Based mode without selecting class → Error message

### Data Validation

- [x] CSV row count matches total points
- [x] CSV selected count matches plot legend
- [x] CSV rejected count matches plot legend
- [x] Mahalanobis distances in CSV match plot positions
- [x] Selected=True samples are within ellipse boundary
- [x] Selected=False samples are outside ellipse boundary

### File Quality

- [x] EPS file opens in vector editor (Inkscape, Illustrator)
- [x] EPS file scales without quality loss
- [x] PNG file displays correctly at 300 DPI
- [x] PDF file embedded in LaTeX compiles correctly
- [x] SVG file editable in vector graphics software

### Special Cases

- [x] Dataset with special characters in name
- [x] Class name with spaces (e.g., "aquarium fish")
- [x] Class name with forward slash (e.g., GTSRB traffic signs)
- [x] Very large dataset (>100K points) → CSV file size reasonable
- [x] Network drive (Google Drive) → Files save correctly
- [x] Multiple saves → Files overwritten correctly

## Benefits

### For Users

1. **Publication-Ready Plots**: EPS/PDF formats suitable for academic papers
2. **Data Portability**: CSV files for external analysis (R, MATLAB, Excel)
3. **Reproducibility**: Save exact filter parameters and results
4. **Flexibility**: Multiple format options for different use cases
5. **Documentation**: Plots serve as visual record of filtering decisions

### For Analysis

1. **Offline Analysis**: CSV data for custom visualizations
2. **Quality Metrics**: Exact Mahalanobis distances for each sample
3. **Verification**: Cross-check selected/rejected samples
4. **Debugging**: Identify borderline samples near threshold
5. **Comparison**: Compare filtering results across datasets/classes

## Future Enhancements (Optional)

### Potential Additions

1. **Metadata JSON**: Save filter parameters alongside plot/CSV
   ```json
   {
     "dataset": "cifar10",
     "mode": "Global",
     "weighted_percentile": 95.0,
     "consistency_percentile": 95.0,
     "keep_percentile": 30.0,
     "ideal_point": [0.85, 0.92],
     "threshold": 2.5,
     "samples_selected": 15000,
     "samples_rejected": 35000
   }
   ```

2. **Batch Export**: "Save All Classes" button for Class-Based mode
3. **DPI Selector**: User-configurable resolution (150, 300, 600)
4. **Custom Filename Prefix**: Allow user to specify prefix
5. **Timestamp Option**: Add timestamp to filenames (optional)
6. **Compression**: ZIP archive for plot + CSV + metadata

## Conclusion

The implementation provides a complete solution for saving Mahalanobis filter results with:
- ✅ Minimal code changes (~200 lines total)
- ✅ Non-breaking (existing functionality unchanged)
- ✅ User-friendly (simple dropdown + button interface)
- ✅ Comprehensive (both plot and data export)
- ✅ Publication-ready (EPS format with 300 DPI)
- ✅ Well-documented (console messages, status updates)
- ✅ Error-resistant (validation and error handling)
