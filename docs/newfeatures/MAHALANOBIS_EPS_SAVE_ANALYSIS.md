# Mahalanobis Filter EPS Format Saving - Code Analysis

## Overview

This document analyzes the Mahalanobis filter visualization code in the interactive data curation system to prepare for adding EPS (Encapsulated PostScript) format saving capability.

## Current Implementation

### File Location
`/workspaces/maveric/maveric/visualization/interactive.py`

### Key Methods

#### 1. `_plot_mahalanobis_analysis()` - Global Mode Plotting
**Location**: Lines 2015-2098
**Purpose**: Plots Mahalanobis distance analysis for global (all classes) filtering

**Current Behavior**:
- Creates a 10×8 inch figure with scatter plot and ellipse
- Shows rejected samples (gray) and selected samples (green)
- Plots ideal point (red star) at user-specified percentiles
- Draws Mahalanobis distance ellipse boundary
- Displays correlation coefficient (ρ)
- **Current save method**: `plt.show()` only (line 2098) - **no file saving**

**Plot Components**:
```python
fig, ax_main = plt.subplots(figsize=(10, 8))

# Scatter plots
ax_main.scatter(all_weighted[~selected_mask], all_consistency[~selected_mask],
               c='gray', alpha=0.3, s=20, label=f'Rejected ({(~selected_mask).sum():,})')
ax_main.scatter(all_weighted[selected_mask], all_consistency[selected_mask],
               c='green', alpha=0.7, s=20, label=f'Selected ({selected_mask.sum():,})')

# Ideal point
ax_main.scatter(ideal_point[0], ideal_point[1],
               c='red', marker='*', s=300, label='Ideal Point',
               edgecolors='darkred', linewidth=1.5, zorder=10)

# Mahalanobis ellipse
ellipse = Ellipse(xy=ideal_point, width=width, height=height,
                 angle=angle, edgecolor='red', facecolor='none',
                 linewidth=2, linestyle='--', label='Selection Boundary')
ax_main.add_patch(ellipse)

# Formatting
ax_main.set_xlabel('Weighted Class Score', fontsize=11)
ax_main.set_ylabel('Consistency', fontsize=11)
ax_main.set_title('Joint Distribution with Mahalanobis Selection Boundary',
                 fontsize=12, fontweight='bold', pad=10)
ax_main.grid(True, alpha=0.3)
ax_main.legend(loc='upper right', fontsize=9)

# Correlation text
ax_main.text(0.02, 0.98, f'ρ = {correlation:.3f}',
            transform=ax_main.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()  # ← Only displays, no saving
```

#### 2. `_plot_mahalanobis_analysis_class_based()` - Class-Based Mode Plotting
**Location**: Lines 2248-2318
**Purpose**: Plots Mahalanobis distance analysis for a specific class

**Current Behavior**:
- Similar structure to global mode
- Title includes class name: `f'Class: {class_name} - Mahalanobis Selection'`
- **Current save method**: `plt.show()` only (line 2318) - **no file saving**

**Differences from Global Mode**:
```python
# Title includes class name
ax_main.set_title(f'Class: {class_name} - Mahalanobis Selection',
                 fontsize=12, fontweight='bold', pad=10)
```

#### 3. `_create_mahalanobis_tab()` - GUI Tab Creation
**Location**: Lines 1306-1750+ (large method)

**GUI Components**:
- **Mode selector**: RadioButtons ('Global' / 'Class-Based')
- **Class selector**: Dropdown (for Class-Based mode)
- **Percentile inputs**:
  - `weighted_percentile_text`: Ideal point for weighted_class_score (default: 95.0)
  - `consistency_percentile_text`: Ideal point for consistency (default: 95.0)
  - `keep_percentile_text`: Percentage to keep (default: 30.0)
  - `keep_count_text`: Exact count to keep (alternative to percentile)
- **Buttons**:
  - `apply_button`: "Apply Filter" - triggers filtering and plotting
  - `add_data_button`: "Add Data" - for class-based mode to accumulate classes
  - `save_filtered_button`: "Save Filtered Data" - saves grid PNG visualizations
  - `reset_button`: "Reset" - clears filter
- **Output widget**: `plot_output` - displays matplotlib plots

**Apply Button Callback**: Lines 1494-1630+
```python
def on_apply_clicked(b):
    with plot_output:
        clear_output(wait=True)
        try:
            mode = mode_selector.value
            keep_percentage = keep_percentile_text.value
            keep_count = keep_count_text.value
            weighted_pct = weighted_percentile_text.value
            consistency_pct = consistency_percentile_text.value

            # ... filtering logic ...

            if mode == 'Global':
                # Apply global filter
                self._apply_mahalanobis_filter(...)
                # Plot results
                self._plot_mahalanobis_analysis()
            else:
                # Apply class-based filter
                self._apply_mahalanobis_filter_class_based(...)
                # Plot results
                self._plot_mahalanobis_analysis_class_based(class_name)
```

### Data Storage Structure

**Filter Information Storage**:
- **Global mode**: `self.mahalanobis_filter_info` (dict)
- **Class-based mode**: `self.mahalanobis_filter_info_class` (dict)

**Stored Information** (both modes):
```python
{
    'ideal_point': np.array([weighted_ideal, consistency_ideal]),
    'covariance': np.array([[var_w, cov], [cov, var_c]]),
    'covariance_inv': np.linalg.inv(covariance),
    'threshold': float,  # Mahalanobis distance threshold
    'correlation': float,  # Pearson correlation coefficient
    'all_samples': {
        'weighted': np.array,
        'consistency': np.array,
        'distances': np.array,
        'data_matrix': np.array
    },
    'selected_mask': np.array(bool),
    'keep_percentile': float,
    'keep_count': int,
    'weighted_percentile': float,
    'consistency_percentile': float,
    # Class-based only:
    'class_name': str  # Only in class-based mode
}
```

## Proposed EPS Saving Implementation

### Option 1: Add "Save Plot" Button (Recommended)

**Advantages**:
- Non-intrusive - doesn't change existing workflow
- User controls when to save
- Can save multiple formats (EPS, PNG, PDF, SVG)

**Implementation Strategy**:

1. **Add new button** to Mahalanobis tab:
   ```python
   save_plot_button = widgets.Button(
       description='Save Plot',
       button_style='info',
       icon='download',
       layout=widgets.Layout(width='120px')
   )
   ```

2. **Add format selector** dropdown:
   ```python
   format_selector = widgets.Dropdown(
       options=['EPS', 'PNG', 'PDF', 'SVG'],
       value='EPS',
       description='Format:',
       layout=widgets.Layout(width='150px'),
       style={'description_width': '50px'}
   )
   ```

3. **Create save callback**:
   ```python
   def on_save_plot_clicked(b):
       if not hasattr(self, 'mahalanobis_filter_info') or \
          (mode_selector.value == 'Class-Based' and not hasattr(self, 'mahalanobis_filter_info_class')):
           print("❌ No plot to save. Apply filter first.")
           return

       # Determine output directory
       if self.data_path.endswith('/raw'):
           base_dir = os.path.dirname(self.data_path)
       else:
           base_dir = self.data_path

       results_dir = os.path.join(base_dir, 'curationResults')
       os.makedirs(results_dir, exist_ok=True)

       # Generate filename
       mode = mode_selector.value
       file_format = format_selector.value.lower()

       if mode == 'Global':
           filename = f"{self.dataset_name}_mahalanobis_global.{file_format}"
       else:
           class_name = self.mahalanobis_filter_info_class.get('class_name', 'unknown')
           safe_class_name = class_name.replace('/', '_').replace('\\', '_')
           filename = f"{self.dataset_name}_mahalanobis_{safe_class_name}.{file_format}"

       filepath = os.path.join(results_dir, filename)

       # Re-create the plot and save it
       fig = self._create_mahalanobis_figure(mode, class_name if mode == 'Class-Based' else None)
       fig.savefig(filepath, format=file_format, bbox_inches='tight', dpi=300)
       plt.close(fig)

       print(f"✅ Plot saved: {filepath}")
   ```

4. **Refactor plotting code** to return figure instead of showing:
   ```python
   def _create_mahalanobis_figure(self, mode='Global', class_name=None):
       """Create Mahalanobis plot figure without displaying it"""
       if mode == 'Global':
           info = self.mahalanobis_filter_info
       else:
           info = self.mahalanobis_filter_info_class

       # Extract data from info dict
       ideal_point = info['ideal_point']
       covariance = info['covariance']
       correlation = info['correlation']
       all_samples = info['all_samples']
       selected_mask = info['selected_mask']

       # Create figure
       fig, ax_main = plt.subplots(figsize=(10, 8))

       # ... (same plotting code as current implementation) ...

       plt.tight_layout()
       return fig  # Return instead of plt.show()
   ```

5. **Update existing plot methods** to use the new helper:
   ```python
   def _plot_mahalanobis_analysis(self):
       """Plot Mahalanobis distance analysis with scatter plot and ellipse"""
       if not self.mahalanobis_filter_info:
           print("❌ No Mahalanobis filter info available")
           return

       fig = self._create_mahalanobis_figure(mode='Global')
       plt.show()

   def _plot_mahalanobis_analysis_class_based(self, class_name):
       """Plot Mahalanobis distance analysis for a specific class"""
       if not hasattr(self, 'mahalanobis_filter_info_class') or not self.mahalanobis_filter_info_class:
           print("❌ No class-based Mahalanobis filter info available")
           return

       fig = self._create_mahalanobis_figure(mode='Class-Based', class_name=class_name)
       plt.show()
   ```

### Option 2: Auto-Save on Apply (Alternative)

**Advantages**:
- Automatic - no extra user action needed
- Consistent with existing "Save Filtered Data" button behavior

**Disadvantages**:
- Creates files without explicit user request
- May clutter output directory

**Implementation Strategy**:

Add to end of `_plot_mahalanobis_analysis()` and `_plot_mahalanobis_analysis_class_based()`:

```python
# At end of _plot_mahalanobis_analysis() (line 2098):
plt.tight_layout()

# Auto-save EPS before showing
if self.data_path:
    if self.data_path.endswith('/raw'):
        base_dir = os.path.dirname(self.data_path)
    else:
        base_dir = self.data_path

    results_dir = os.path.join(base_dir, 'curationResults')
    os.makedirs(results_dir, exist_ok=True)

    filename = f"{self.dataset_name}_mahalanobis_global.eps"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, format='eps', bbox_inches='tight', dpi=300)
    print(f"📊 Plot saved: {filepath}")

plt.show()
```

## EPS Format Specifics

### Matplotlib EPS Support

Matplotlib natively supports EPS format via `plt.savefig()`:

```python
# Basic EPS save
plt.savefig('filename.eps', format='eps')

# High-quality EPS save
plt.savefig('filename.eps',
            format='eps',
            bbox_inches='tight',  # Trim whitespace
            dpi=300,              # High resolution
            transparent=False)    # White background
```

### EPS Format Advantages
- **Vector format**: Scales infinitely without quality loss
- **Publication-ready**: Accepted by most journals and conferences
- **Text preservation**: Text remains editable and searchable
- **Small file size**: Efficient for plots with few data points
- **LaTeX compatible**: Direct inclusion in LaTeX documents

### EPS Format Considerations
- **Large scatter plots**: EPS files can be large with thousands of points
  - Mahalanobis plots typically have 1,000-50,000 points
  - Consider rasterizing scatter plots while keeping ellipse/text as vector
- **Font embedding**: May need to configure font handling for consistency
- **Modern alternatives**: PDF and SVG are more modern vector formats

### Recommended EPS Settings

```python
# For publication-quality plots
save_kwargs = {
    'format': 'eps',
    'bbox_inches': 'tight',
    'dpi': 300,
    'transparent': False,
    'pad_inches': 0.1
}

# For large scatter plots (optional rasterization)
save_kwargs_raster = {
    'format': 'eps',
    'bbox_inches': 'tight',
    'dpi': 300,
    'rasterized': True  # Rasterize scatter points, keep vector text/lines
}
```

## Implementation Checklist

### Minimal Implementation (EPS only, auto-save)
- [ ] Modify `_plot_mahalanobis_analysis()` to save EPS after plotting
- [ ] Modify `_plot_mahalanobis_analysis_class_based()` to save EPS after plotting
- [ ] Generate appropriate filenames (global vs class-based)
- [ ] Create output directory if needed
- [ ] Add console message confirming save

### Recommended Implementation (Save button with format selector)
- [ ] Create `_create_mahalanobis_figure()` helper method
- [ ] Refactor `_plot_mahalanobis_analysis()` to use helper
- [ ] Refactor `_plot_mahalanobis_analysis_class_based()` to use helper
- [ ] Add "Save Plot" button to GUI
- [ ] Add format selector dropdown (EPS, PNG, PDF, SVG)
- [ ] Create `on_save_plot_clicked()` callback
- [ ] Wire up button click event
- [ ] Test with both Global and Class-Based modes
- [ ] Add error handling for missing filter info
- [ ] Document in user guide

### Advanced Features (Optional)
- [ ] Add DPI selector (150, 300, 600)
- [ ] Add transparent background option
- [ ] Add filename prefix/suffix customization
- [ ] Batch save all classes in Class-Based mode
- [ ] Save metadata JSON alongside plot (filter settings, statistics)

## File Naming Conventions

### Current Convention (Grid PNGs)
Format: `{dataset_name}_{class_name}_{sequence}.png`
Example: `cifar10_airplane_001.png`

### Proposed Convention (Mahalanobis Plots)

**Global mode**:
```
{dataset_name}_mahalanobis_global.{ext}
Example: cifar10_mahalanobis_global.eps
```

**Class-based mode**:
```
{dataset_name}_mahalanobis_{class_name}.{ext}
Example: cifar10_mahalanobis_airplane.eps
Example: cifar100_mahalanobis_aquarium_fish.eps  (spaces replaced with underscores)
```

**With metadata**:
```
{dataset_name}_mahalanobis_{mode}_{timestamp}.{ext}
Example: cifar10_mahalanobis_global_20260322_143022.eps
```

## Output Directory Structure

```
{base_dir}/
├── raw/                          # Original retrieval results
│   ├── cifar10_rotation_001.json
│   └── ...
├── images/                       # Cached images
│   ├── sample_001.jpg
│   └── ...
└── curationResults/              # All curation outputs ← Target directory
    ├── cifar10_grid_001.png      # Existing: sample grids
    ├── cifar10_grid_002.png
    ├── cifar10_airplane_001.png  # Existing: class-based grids
    ├── cifar10_mahalanobis_global.eps     # NEW: global Mahalanobis plot
    ├── cifar10_mahalanobis_airplane.eps   # NEW: class Mahalanobis plot
    └── training_rotation_001.json         # Existing: training data
```

## Related Code Sections

### Existing Image Saving (for reference)
**Method**: `_save_class_filtered_grids()` (lines 2320-2434)
- Saves 10×10 grids as PNG
- Uses `os.makedirs(results_dir, exist_ok=True)`
- Filename format: `f"{self.dataset_name}_{safe_class_name}_{grid_idx+1:03d}.png"`
- Save call: `plt.savefig(grid_path, dpi=150, bbox_inches='tight')`

### Plot Output Widget
- Created at line 1451: `plot_output = widgets.Output()`
- Used in apply callback: `with plot_output: clear_output(wait=True); ...`
- This is where `plt.show()` displays - **cannot directly save from here**

## Testing Plan

1. **Global mode EPS save**:
   - Load CIFAR-10 dataset
   - Apply global Mahalanobis filter (30% keep)
   - Click "Save Plot" → verify EPS created in curationResults/
   - Open EPS in vector editor (Inkscape, Illustrator) → verify vector quality

2. **Class-based mode EPS save**:
   - Select "airplane" class
   - Apply class filter
   - Click "Save Plot" → verify class-specific EPS created

3. **Format selector**:
   - Test all formats: EPS, PNG, PDF, SVG
   - Verify correct file extensions
   - Verify quality at different DPI settings

4. **Edge cases**:
   - Click "Save Plot" before applying filter → error message
   - Save with special characters in class name → sanitized filename
   - Save when output directory doesn't exist → auto-create

## Conclusion

The codebase is well-structured for adding EPS saving capability. The recommended approach is:

1. **Refactor plotting code** into a `_create_mahalanobis_figure()` helper that returns a figure object
2. **Add GUI button** with format selector for user-controlled saving
3. **Use standard matplotlib** `savefig()` with EPS format
4. **Follow existing conventions** for output directory and filename structure

This approach is:
- **Minimal code change**: ~50-100 lines total
- **Non-breaking**: Existing functionality unchanged
- **User-friendly**: Explicit save action with format choice
- **Consistent**: Matches existing grid saving pattern
