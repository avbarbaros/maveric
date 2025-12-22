# Mahalanobis Filter - Reset Button Feature

**Date**: December 22, 2025  
**Feature**: Reset button for clearing filtered data  
**Status**: ✅ Implemented

---

## Overview

Added a "Reset" button to the Mahalanobis Filter tab that clears filtered data and returns to the original state. The behavior is mode-dependent: Global mode restores pre-filter data, while Class-Based mode can clear specific classes or all accumulated data.

---

## Button Properties

```python
reset_button = widgets.Button(
    description='Reset',
    button_style='danger',    # Red color indicating destructive action
    icon='undo',              # Undo icon
    layout=widgets.Layout(width='100px')
)
```

**Location in UI**: Right side of button row, after Apply/Add Data/Save Filtered Data buttons

---

## Behavior

### Global Mode

**Action**: Restore data before Mahalanobis filter was applied

**Logic**:
```python
if self.data_before_mahalanobis is not None:
    self.filtered_data = self.data_before_mahalanobis.copy()
    self.data_before_mahalanobis = None  # Clear backup
```

**Console Output**:
```
🔄 Reset to data before Mahalanobis filter
   Total samples: 50,000
✅ Global filter reset successfully
   Restored 50,000 samples
```

**Use Case**: User applied Mahalanobis filter, wants to undo and start over

---

### Class-Based Mode (No Class Selected)

**Action**: Clear ALL accumulated class-based filtered data

**Logic**:
```python
if selected_class == 'Select class...':
    if self.class_based_filtered_data:
        num_classes = len(self.class_based_filtered_data)
        self.class_based_filtered_data.clear()
```

**Console Output**:
```
🔄 Cleared all class-based filtered data (3 classes)
✅ All class-based data cleared
   Removed 3 classes
```

**Use Case**: User added several classes, wants to start over from scratch

---

### Class-Based Mode (Specific Class Selected)

**Action**: Remove filtered data for selected class only

**Logic**:
```python
if selected_class in self.class_based_filtered_data:
    sample_count = len(self.class_based_filtered_data[selected_class])
    del self.class_based_filtered_data[selected_class]
    
    # Re-consolidate remaining data
    if self.class_based_filtered_data:
        self._consolidate_class_based_data()
    else:
        # No more class data - restore original
        if self.data_before_mahalanobis is not None:
            self.filtered_data = self.data_before_mahalanobis.copy()
```

**Console Output**:
```
🔄 Cleared filtered data for class 'airplane' (200 samples)
✅ Class 'airplane' data cleared
   Remaining classes: 2
```

**Use Case**: User is unhappy with filtering for one specific class, wants to re-filter it with different settings

---

## Implementation Details

### Callback Function

**Location**: [interactive.py:1557-1623](maveric/visualization/interactive.py#L1557-L1623)

**Key Features**:
- Mode detection via `mode_selector.value`
- Class selection detection via `class_selector.value`
- Smart data restoration (backup → filtered_data)
- Automatic re-consolidation when removing specific class
- Clear status messages for all scenarios

### Button Registration

**Location**: [interactive.py:1628](maveric/visualization/interactive.py#L1628)

```python
reset_button.on_click(on_reset_clicked)
```

### Layout Integration

**Location**: [interactive.py:1638-1642](maveric/visualization/interactive.py#L1638-L1642)

```python
widgets.HBox([
    apply_button,
    add_data_button,
    save_filtered_button,
    reset_button           # ← Added here
], layout=widgets.Layout(margin='5px 0')),
```

---

## Usage Examples

### Example 1: Global Mode Reset

```python
# User applies global filter
# 50,000 samples → 15,000 samples (30%)

# User realizes they want to try different percentiles
# Click "Reset" button
# → 15,000 samples → 50,000 samples (restored)

# Can now re-apply with new settings
```

### Example 2: Class-Based Clear All

```python
# User filters 3 classes:
#   airplane: 200 samples
#   automobile: 150 samples
#   bird: 180 samples

# User wants to start over completely
# Class selector: "Select class..."
# Click "Reset" button
# → All 3 classes removed

# Can start fresh with different approach
```

### Example 3: Class-Based Clear Specific

```python
# User has filtered 3 classes:
#   airplane: 200 samples
#   automobile: 150 samples
#   bird: 180 samples

# User unhappy with "airplane" filtering
# Class selector: "airplane"
# Click "Reset" button
# → Only "airplane" removed (200 samples)
# → Remaining: automobile (150) + bird (180) = 330 samples
# → Data automatically re-consolidated

# Can re-filter "airplane" with new settings
```

---

## Edge Cases Handled

### 1. No Filter Applied (Global)
```
Click Reset → "ℹ️  No previous Mahalanobis filter applied"
```

### 2. No Class Data (Class-Based, no selection)
```
Click Reset → "ℹ️  No class-based filtered data to clear"
```

### 3. Class Not Filtered (Class-Based, specific class)
```
Select "cat" → Click Reset → "ℹ️  No filtered data for class 'cat'"
```

### 4. Last Class Removed (Class-Based)
```
Only 1 class left → Remove it → Restores original data from backup
```

---

## Benefits

✅ **Flexible**: Works in both Global and Class-Based modes  
✅ **Intelligent**: Mode-aware behavior (global vs per-class)  
✅ **Safe**: Danger button style warns of destructive action  
✅ **Granular**: Can clear all data or specific class  
✅ **Automatic**: Re-consolidates data after removing specific class  
✅ **Informative**: Clear console messages explain what happened  

---

## Visual Layout

```
┌────────────────────────────────────────────────────────┐
│ Mode: ○ Global  ○ Class-Based                          │
│ Class: [airplane ▼]                                    │
├────────────────────────────────────────────────────────┤
│ Weighted %ile: [95.0]  Consistency %ile: [95.0]       │
│ Keep %ile: [30.0]                                      │
├────────────────────────────────────────────────────────┤
│ [Apply Filter] [Add Data] [Save Filtered] [Reset] ←NEW│
├────────────────────────────────────────────────────────┤
│ Status: Configure percentiles and click Apply          │
└────────────────────────────────────────────────────────┘
```

---

## Testing

**Manual Test Checklist**:
- [ ] Global mode: Apply filter → Reset → Verify data restored
- [ ] Class-Based: Add 3 classes → Reset (no selection) → Verify all cleared
- [ ] Class-Based: Add 3 classes → Select one → Reset → Verify only that class cleared
- [ ] Class-Based: Last class → Reset → Verify original data restored
- [ ] Edge case: Reset with no filter → Verify friendly message
- [ ] Edge case: Reset with no class data → Verify friendly message

**Code Structure Test**: ✅ PASSED
- Reset button created with correct properties
- Callback defined and registered
- Button included in layout

---

## Documentation

- **Main Guide**: [MAHALANOBIS_CLASS_BASED_MODE.md](MAHALANOBIS_CLASS_BASED_MODE.md) (Section 6)
- **Implementation**: [interactive.py:1381-1386](maveric/visualization/interactive.py#L1381-L1386) (button creation)
- **Callback**: [interactive.py:1557-1623](maveric/visualization/interactive.py#L1557-L1623) (reset logic)

---

**Summary**: The Reset button provides a safe, intelligent way to undo Mahalanobis filtering and start over, with mode-aware behavior that handles both global and per-class workflows. 🔄
