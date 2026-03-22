# Quick Guide: Saving Mahalanobis Plots

## 🚀 Quick Start (3 Steps)

### 1. Apply Filter
- Go to **Tab 3: Mahalanobis Filter**
- Set your percentiles
- Click **"Apply Filter"**
- Plot appears below ✓

### 2. Select Format
- Choose from dropdown: **EPS** (recommended), PNG, PDF, or SVG

### 3. Save
- Click **"Save Plot"** button
- Done! ✅

## 📁 Where Are Files Saved?

```
{your_data_dir}/curationResults/
├── {dataset}_mahalanobis_global.eps          ← Plot file
├── {dataset}_mahalanobis_global_data.csv     ← Data file
```

**Example**:
```
./results/cifar10/curationResults/
├── cifar10_mahalanobis_global.eps
├── cifar10_mahalanobis_global_data.csv
```

## 📊 What Gets Saved?

### Plot File (`.eps`, `.png`, `.pdf`, or `.svg`)
- High-quality scatter plot
- Green dots = selected samples
- Gray dots = rejected samples
- Red star = ideal point
- Red ellipse = selection boundary
- Correlation coefficient (ρ) displayed

### Data File (`.csv`)
Always saved alongside plot:

```csv
weighted_class_score,consistency,mahalanobis_distance,selected
0.850000,0.920000,1.234567,True
0.830000,0.910000,1.456789,True
...
0.450000,0.720000,8.901234,False
```

**Columns**:
- `weighted_class_score` - Quality metric (X-axis)
- `consistency` - Consistency metric (Y-axis)
- `mahalanobis_distance` - Distance from ideal point
- `selected` - True = kept, False = rejected

## 🎨 Format Guide

| Format | When to Use | File Size |
|--------|-------------|-----------|
| **EPS** | 📄 Publications, papers, LaTeX | ~50-500 KB |
| **PDF** | 📤 Sharing, presentations | ~50-500 KB |
| **PNG** | 👁️ Quick view, slides | ~200-800 KB |
| **SVG** | 🌐 Web, editing in Inkscape | ~50-500 KB |

**Recommendation**: Use **EPS** for academic papers, **PDF** for general use.

## 💡 Tips

### Global Mode
```python
1. Mode: Global
2. Set percentiles (e.g., 95, 95, 30)
3. Apply Filter
4. Format: EPS
5. Save Plot
```
**Output**: `cifar10_mahalanobis_global.eps` + `.csv`

### Class-Based Mode
```python
1. Mode: Class-Based
2. Class: airplane
3. Set percentiles
4. Apply Filter
5. Format: EPS
6. Save Plot
```
**Output**: `cifar10_mahalanobis_airplane.eps` + `.csv`

### Multiple Classes
Apply filter and save for each class:
```python
For each class:
  - Select class
  - Apply Filter
  - Save Plot
```

## ❓ Common Questions

**Q: Can I save without applying filter?**
A: No, you'll get an error. Apply filter first.

**Q: Can I change format after saving?**
A: Yes! Select new format and click "Save Plot" again.

**Q: Where's the CSV file?**
A: Same directory as plot, with `_data.csv` suffix.

**Q: Can I open CSV in Excel?**
A: Yes! Standard CSV format, compatible with Excel, R, Python, MATLAB.

**Q: What if filename has special characters?**
A: Automatically replaced with underscores (e.g., `aquarium fish` → `aquarium_fish`).

## 🔧 Troubleshooting

**Error: "No filter info available"**
→ Click "Apply Filter" first

**Error: "Select a class first"**
→ Choose class from dropdown (Class-Based mode)

**Files not appearing**
→ Check console for exact path
→ Verify `curationResults/` folder exists

**Plot looks wrong**
→ Re-apply filter
→ Check percentile values are correct

## 📖 More Information

- **Full Documentation**: See `MAHALANOBIS_SAVE_IMPLEMENTATION.md`
- **Analysis Details**: See `MAHALANOBIS_EPS_SAVE_ANALYSIS.md`
- **Implementation Summary**: See `IMPLEMENTATION_SUMMARY.md`
- **Test Script**: Run `python test_mahalanobis_save.py`

## ✅ Checklist

Before saving plots:
- [ ] Applied Mahalanobis filter
- [ ] Plot looks correct
- [ ] Selected appropriate format
- [ ] Know where files will be saved

After saving:
- [ ] Check console for confirmation messages
- [ ] Verify plot file exists
- [ ] Verify CSV file exists
- [ ] Open CSV to check data

That's it! Happy plotting! 🎉
