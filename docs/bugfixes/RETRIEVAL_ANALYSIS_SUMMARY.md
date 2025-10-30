# MAVERIC Data Retrieval Process - Executive Summary

## Critical Issues Found

### Issue 1: O(n×c) Complexity in Reference Sample Selection (BLOCKER)
**File:** `maveric/datasets/elevater_datasets.py` line 477-530
**Method:** `_get_torchvision_reference_samples()`

**Problem:**
```python
for class_idx, class_name in enumerate(self.class_names):
    # This re-iterates entire dataset for EACH class!
    class_indices = [i for i, (_, label) in enumerate(self._dataset) if label == class_idx]
```

**Impact:**
- Food101: 75,750 × 101 = **7.6 million iterations**
- With Google Drive NFS: **15-30+ minutes** for reference prep alone
- Process appears "stuck" because no progress output for long periods

**Evidence:**
- Commit 8394e83: "Data retrieval process stuck" → added logs
- Commit f8313c8: "Performance improvement" → optimized to O(n) 
- Commit b4fe424: "Log added for debugging" → still having issues
- Current code has O(n×c) pattern again = **REGRESSION**

**Solution:**
Implement single-pass index building:
```python
class_indices_map = {class_idx: [] for class_idx in range(len(self.class_names))}
for i in range(len(self._dataset)):
    _, label = self._dataset[i]
    class_indices_map[label].append(i)  # O(n) - just once

# Then process classes using pre-built map
for class_idx, class_name in enumerate(self.class_names):
    class_indices = class_indices_map.get(class_idx, [])  # Fast lookup
```

---

### Issue 2: Aggressive Network Timeout
**File:** `maveric/retrieval/cache_manager.py` line 246
**Method:** `download_and_cache_image()`

**Problem:**
```python
response = requests.get(url, timeout=5)  # 5 seconds is too short
```

**Impact:**
- Large images (>10MB) will timeout with slow networks
- Retry 3 times = 15+ seconds per image
- On slow networks, most/all images fail
- Cascading failures through entire dataset

**Solution:**
- Increase to 15-30 seconds
- Use adaptive backoff: exponential delay between retries
- Or make configurable via config

---

### Issue 3: JSON Export File I/O Not Timeout-Protected
**File:** `maveric/retrieval/retriever.py` line 208-242
**Method:** `_export_rotation_file()`

**Problem:**
```python
with open(filepath, 'w') as f:
    json.dump(batch, f, indent=2)  # No timeout, can block indefinitely
```

**Impact:**
- Large JSON files (50-100MB) on network filesystem can hang
- `indent=2` increases file size by ~50%
- Can appear completely stuck when rotation batch is exported
- No error recovery

**Solution:**
- Write to temp file first, then atomic rename
- Add timeout using signal handler or threading
- Remove `indent=2` or use compact JSON
- Add error recovery with retry logic

---

### Issue 4: EfficientNet Enabled by Default (Performance)
**File:** `maveric/config.py` line 89
**Config:** `enable_target_class_quality: bool = True`

**Problem:**
- EfficientNet adds 50-70% overhead (300-600s for 1000 samples)
- Runs on CPU even if GPU available
- Default enabled but not well-documented
- Most users don't expect 20-30 minute retrieval times

**Solution:**
- Change default to `False`
- Or at minimum, add warning about performance impact
- Already has `--disable-efficientnet` flag for CLI

---

### Issue 5: Weak Cache Validation
**File:** `maveric/retrieval/retriever.py` line 118-138
**Method:** `prepare_reference_embeddings()`

**Problem:**
```python
if ref_cache and text_cache:  # Just checks if non-empty dict
    # Doesn't validate embeddings shape, dtype, or values
```

**Impact:**
- Corrupted cache might be silently used
- Causes confusing errors downstream
- No cache invalidation when CLIP model changes

**Solution:**
- Validate embeddings shape matches expected
- Check dtype is numpy array/tensor
- Add CLIP model version to cache name

---

### Issue 6: No Progress Logging During Long Operations
**Files:** 
- `maveric/retrieval/retriever.py` line 456
- `maveric/retrieval/cache_manager.py` 

**Problem:**
- CLIP model loading (10-30s): no progress output
- Reference embedding generation: no per-step progress
- Network timeouts: failures logged but not aggregated
- User sees nothing for minutes, thinks it's hung

**Solution:**
- Add progress bars for model loading
- Log incremental progress during embedding generation
- Summarize network failures per batch

---

## Impact Assessment

### Severity: CRITICAL
- **Reference Selection**: 15-30 min blocking time
- **Network Timeout**: Can fail entire dataset
- **File Write Blocking**: Can hang indefinitely

### Scope: Data Retrieval Pipeline
- Affects all datasets, especially large ones (Food101, Country211)
- Affects all users on slow networks or Google Drive
- Affects all users by default with EfficientNet overhead

### Reproduction Steps:
1. Run: `python experiments/01_data_retrieval.py --config maveric_config.yaml`
2. Select Food101 or Country211 (large datasets)
3. Press Enter to use defaults
4. Observe: Process appears to hang for 15-30+ minutes

---

## Quick Fixes (Priority Order)

### High Priority (1-2 hours)
1. **Re-implement O(n) reference sample selection** (commit f8313c8 showed the fix)
2. **Increase network timeout** to 15-30 seconds
3. **Add atomic file writes** (temp file + rename)
4. **Add progress logging** for long operations

### Medium Priority (1-2 hours)
5. **Change EfficientNet default to disabled**
6. **Add cache validation**
7. **Implement connection pooling** for requests

### Low Priority (follow-up)
8. **Add circuit breaker** for repeated failures
9. **Improve thread safety** in console output
10. **Add checkpointing** for crash recovery

---

## Files to Review
- `/workspaces/maveric/maveric/datasets/elevater_datasets.py` (lines 456-530)
- `/workspaces/maveric/maveric/retrieval/cache_manager.py` (lines 225-277)
- `/workspaces/maveric/maveric/retrieval/retriever.py` (lines 208-242, 278-426)
- `/workspaces/maveric/experiments/01_data_retrieval.py`

## Related Commits
- b4fe424: Log added for blocked retrieval process debugging
- f8313c8: Performance improvement at data retrieval process
- 8394e83: Data retrieval process stuck following logs added

