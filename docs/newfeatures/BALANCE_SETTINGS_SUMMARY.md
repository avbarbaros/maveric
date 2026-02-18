# Balance Settings Tab - Quick Reference

**Last Updated**: December 21, 2025

---

## What Changed? (December 21, 2025)

| Change | Before | After | Benefit |
|--------|--------|-------|---------|
| **Min Samples** | IntSlider (1-100) | Removed (hardcoded to 1) | Simpler UI, all classes kept |
| **Oversampling** | Checkbox (default width) | Checkbox (500px width) | Full visibility |
| **Sorting** | Hardcoded to consistency | Dropdown (Consistency/Weighted) | Flexible sample ranking |

---

## Tab Controls

```
┌──────────────────────────────────────────────────────────┐
│ Balance Settings                                         │
├──────────────────────────────────────────────────────────┤
│ Strategy:     [Dropdown: none/min/max/median/mean]      │
│ Sorting:      [Dropdown: Consistency/Weighted] ← NEW    │
│ ☐ Enable Oversampling                                   │
│ [Apply Balance]                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Sorting Options

### Consistency (Default) ⭐
- Ranks samples by `consistency` score
- Best for: General-purpose training
- Ensures: Strong image-text alignment

### Weighted
- Ranks samples by `weighted_class_score`
- Best for: Class-specific quality
- Ensures: Best class representatives

---

## Quick Usage

```python
from maveric.visualization import start_interactive_gui

gui = start_interactive_gui('cifar10')

# Go to Tab 4: Balance Settings
# 1. Select strategy (e.g., "median")
# 2. Select sorting: "Consistency" or "Weighted"
# 3. Check "Enable Oversampling" if needed
# 4. Click "Apply Balance"
```

---

## Strategy Comparison

| Strategy | Target Size | When to Use |
|----------|-------------|-------------|
| `none` | Original | No balancing needed |
| `min` | Smallest class | Avoid duplicates, pure undersampling |
| `max` | Largest class | Maximum data retention, needs oversampling |
| `median` | Median class | Moderate balance, mixed sampling |
| `mean` | Average class | Moderate balance, mixed sampling |

---

## Example Results

### Before Balancing
```
class_a: 1,000 samples
class_b: 5,000 samples
class_c: 3,000 samples
```

### Strategy: `min`, Sorting: `Consistency`, Oversampling: OFF
```
class_a: 1,000 samples (top 1,000 by consistency)
class_b: 1,000 samples (top 1,000 by consistency)
class_c: 1,000 samples (top 1,000 by consistency)
```

### Strategy: `median`, Sorting: `Weighted`, Oversampling: ON
```
class_a: 3,000 samples (top 1,000 + 2,000 duplicates by weighted score)
class_b: 3,000 samples (top 3,000 by weighted score)
class_c: 3,000 samples (all samples kept)
```

---

## Decision Matrix

### Choose Consistency When:
✅ Training general-purpose classifiers
✅ Need reliable cross-modal alignment
✅ Not sure which to use (safe default)

### Choose Weighted When:
✅ Optimizing class-specific quality
✅ Building class-focused validation sets
✅ Prioritizing per-class metrics

---

## Technical Details

### Min Samples = 1 (Hardcoded)
- **Impact**: All classes are kept, no class filtering
- **Previous**: User could set 1-100, classes below threshold removed
- **Current**: min_samples always = 1, no classes removed

### Sorting Implementation
```python
# Sort samples by chosen metric (descending = best first)
if sorting_method in class_data.columns:
    class_data = class_data.sort_values(sorting_method, ascending=False)

# When undersampling: keeps top N samples
# When oversampling: duplicates top samples first
```

### Widget Visibility
```python
# All controls now have explicit 500px width
balance_strategy_widget = widgets.Dropdown(..., layout=widgets.Layout(width='500px'))
balance_sorting_widget = widgets.Dropdown(..., layout=widgets.Layout(width='500px'))
balance_oversampling_widget = widgets.Checkbox(..., layout=widgets.Layout(width='500px'))
```

---

## Common Workflows

### Workflow 1: Clean Balanced Training Set
```
Strategy: min
Sorting: Consistency
Oversampling: OFF

→ All classes same size (smallest class)
→ No artificial duplicates
→ Best cross-modal quality
```

### Workflow 2: Maximum Data with Quality
```
Strategy: max
Sorting: Weighted
Oversampling: ON

→ All classes same size (largest class)
→ Small classes oversampled
→ Best class-specific quality
```

### Workflow 3: Moderate Balance (Recommended)
```
Strategy: median
Sorting: Consistency
Oversampling: ON

→ All classes same size (median)
→ Mixed under/oversampling
→ Balanced cross-modal quality
```

---

## Migration Notes

### For Existing Code

**Old Code** (before Dec 21, 2025):
```python
self.balance_settings = {
    'balance_strategy': 'median',
    'balance_min_samples': 10,  # User-configured
    'balance_enable_oversampling': True
}
```

**New Code** (after Dec 21, 2025):
```python
self.balance_settings = {
    'balance_strategy': 'median',
    'balance_min_samples': 1,  # Hardcoded
    'balance_enable_oversampling': True,
    'balance_sorting_method': 'consistency'  # New parameter
}
```

**Impact**: No breaking changes, backward compatible with `.get()` fallback

---

## Testing

Run the test suite to verify:
```bash
python test_balance_settings_updates.py
```

Expected output: ✅ ALL TESTS PASSED

---

## Documentation

- **Complete Guide**: [BALANCE_SETTINGS_GUIDE.md](BALANCE_SETTINGS_GUIDE.md)
- **This Summary**: [BALANCE_SETTINGS_SUMMARY.md](BALANCE_SETTINGS_SUMMARY.md)
- **Code Changes**: [interactive.py](maveric/visualization/interactive.py)
- **Test Suite**: [test_balance_settings_updates.py](test_balance_settings_updates.py)

---

## Summary

**3 Simple Changes, Big Impact:**

1. ❌ **Removed**: Min Samples slider → Hardcoded to 1
2. ✨ **Enhanced**: Oversampling checkbox → Full visibility (500px)
3. ⭐ **Added**: Sorting dropdown → Consistency (default) or Weighted

**Result**: Cleaner UI + Flexible sample selection! 🎉

---

**Last Updated**: December 21, 2025
