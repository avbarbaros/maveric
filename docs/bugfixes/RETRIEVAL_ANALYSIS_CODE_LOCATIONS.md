# MAVERIC Data Retrieval - Detailed Code Locations and References

## 1. ENTRY POINT

### experiments/01_data_retrieval.py
Main user-facing script for data retrieval

**Key Functions:**
- Line 284: `main()` - Main entry point
- Line 242: `setup_maveric()` - Initialize MAVERIC with config
- Line 150: `get_user_dataset_selection()` - Interactive dataset picker
- Line 119: `get_user_start_index()` - Interactive start index selector
- Line 83: `get_user_file_sequence()` - Interactive file sequence selector
- Line 359: `maveric.retrieve()` - Main retrieval call

**Key Configuration Loading:**
- Line 289: `load_config_file()` - Load YAML config
- Line 307: Read `num_samples` from config
- Line 311: Read `results_dir` from config
- Line 316: Read `enable_target_class_quality` flag

---

## 2. MAIN MAVERIC CLASS

### maveric/main.py

**MAVERIC Class:**
- Line 24-100: Class definition and initialization
- Line 32: `__init__()` - Initialize MAVERIC instance
- Line 43-45: Real-time stats initialization with config flag
- Line 102-170: `retrieve()` - Main retrieval orchestration method

**retrieve() Method Details:**
- Line 127: Log start of retrieval
- Line 130-140: Check cache for cached results
- Line 143-146: Create dataset handler (REACTDatasetHandler)
- Line 149-150: Skip to start_index if needed
- Line 153-162: Call self.retriever.retrieve() with all parameters
- Line 167-168: Finalize real-time stats display

---

## 3. CORE RETRIEVER ENGINE

### maveric/retrieval/retriever.py

**Retriever Class: Lines 26-643**

**Initialization: Lines 35-70**
- Line 37: CLIP model name parameter
- Line 38: Device parameter (cuda/cpu)
- Line 40: Cache manager instance
- Line 63: `enable_target_class_quality` flag
- Line 66: `_init_clip_model()`
- Line 69: `_init_quality_metrics()`

**CLIP Model Loading: Lines 75-85**
- Line 79: `clip.load()` call
- Line 81: Device fallback to CPU

**Quality Metrics Init: Lines 87-100**
- Line 89: Always initialized: ResolutionMetric, SharpnessMetric, ColorDiversityMetric
- Line 96-100: Conditionally initialize TargetClassQualityMetric

**Reference Embeddings Preparation: Lines 102-206**
- Line 115-138: Try to load from cache
- Line 118: Cache name construction
- Line 120: Check if cache exists
- Line 124-133: Lenient validation (WEAK VALIDATION BUG)
- Line 140-147: Load target dataset
- Line 149-151: Get reference samples with seed
- Line 157-169: Create image embeddings
- Line 172-184: Create text embeddings
- Line 187-205: Save to cache

**Main Retrieval Loop: Lines 428-615**
- Line 456: Log "Preparing reference embeddings"
- Line 457: Call `prepare_reference_embeddings()`
- Line 488: Main for loop starting: `for idx, item in enumerate(dataset_handler)`
- Line 490-491: Skip if before start_index
- Line 494-495: Break if enough samples processed
- Line 498-505: Update real-time stats with index
- Line 508-509: Extract URL and TEXT from item
- Line 515: Call `compute_sample_scores()`
- Line 537-540: Add sample to batch and all_samples
- Line 543-547: Update batch position stats
- Line 558-572: Export rotation when batch full (BLOCKING POINT)
- Line 585-592: Export remaining batch at end
- Line 605-615: Create and return RetrievalResult

**Score Computation: Lines 278-426**
- Line 295-303: Download/get cached image
- Line 308-319: Compute CLIP embeddings
- Line 329-346: Compute EfficientNet mappings if enabled (PERFORMANCE BOTTLENECK)
- Line 349-396: Compute similarity scores for each class
- Line 399-417: Compute quality scores

**Rotation File Export: Lines 208-242**
- Line 227: Create export directory
- Line 230: Create filename with dataset name and file_id
- Line 234-235: json.dump() with indent=2 (FILE I/O BLOCKING)

**ImageNet Mapping Computation: Lines 244-276**
- Line 267: Run EfficientNet once per image
- Line 270: Compute mappings for all target classes
- Line 272: Return results

---

## 4. CACHE MANAGER

### maveric/retrieval/cache_manager.py

**CacheManager Class: Lines 17-620**

**Initialization: Lines 25-65**
- Line 42: base_dir as Path
- Line 48-53: Create cache directory references
- Line 55: `_create_directories()` call
- Line 57-65: Initialize stats dictionary

**Cache Directory Structure: Lines 67-82**
- Line 69-75: List of directories to create

**Hierarchical Path: Lines 88-110**
- Line 104: Use first 2 chars of hash as subdir (256 possible directories)
- Line 106-108: Create subdirectories

**Cache Image: Lines 112-152**
- Line 126: Generate URL hash
- Line 130: Get hierarchical cache path
- Line 141-143: Save image with format

**Get Cached Image: Lines 154-223**
- Line 170: Try hierarchical structure first
- Line 198-220: Fallback to flat structure
- Line 222: Mark as cache miss

**Download and Cache: Lines 225-277** ← TIMEOUT BUG
- Line 246: `requests.get(url, timeout=5)` ← 5s TIMEOUT TOO SHORT
- Line 267-268: Sleep 1s before retry
- Line 271: Track failed download

**Save Results: Lines 279-312**
- Line 305-306: `json.dump()` to results file

**Load Results: Lines 314-358**
- Line 336: Find matching files with glob pattern
- Line 345: `json.load()` from file

**Embeddings Cache: Lines 451-513**
- Line 478: Save as .npz file
- Line 505: Load embeddings from .npz

---

## 5. DATASET PREPARATION - CRITICAL BUG LOCATION

### maveric/datasets/elevater_datasets.py

**ELEVATERDataset Class: Lines 16-680**

**Reference Samples: Lines 428-530** ← MAIN BUG HERE
- Line 440-442: Set random seeds
- Line 445-449: Branch between torchvision and file-based

**Torchvision Reference Samples: Lines 456-530** ← O(n×c) COMPLEXITY BUG
- Line 463-466: Print dataset info
- Line 468-490: Build class indices map and print progress (CURRENT CODE - REVERTED)
  - Line 472: Create empty class_indices_map
  - Line 474-488: Iterate through dataset ONCE (O(n)) ✓ CORRECT PART
  - Line 475: progress_interval calculation
  - Line 477-488: Loop through each sample ← THIS IS THE OPTIMIZATION
- Line 492-530: Process each class
  - Line 493: For loop over classes ← STARTS O(n×c) PATTERN
  - **CRITICAL BUG: Lines 494** ← RE-ITERATES ENTIRE DATASET FOR EACH CLASS
    ```python
    class_indices = [i for i, (_, label) in enumerate(self._dataset) if label == class_idx]
    ```
  - Line 504-508: Sample indices using pre-built map (would be fast if indices pre-computed)
  - Line 512-525: Load images for this class

**File-based Reference Samples: Lines 532-680**

---

## 6. REAL-TIME STATISTICS

### maveric/core/progress.py

**RealTimeStats Class: Lines 8-84**

**Initialization: Lines 11-23**
- Line 20: threading.Lock() - Lock for thread safety
- Line 21: last_update - Time of last display
- Line 22: update_interval - Gap between displays (default 2.0s)

**Update Stats: Lines 25-36**
- Line 27: Acquire lock
- Line 29: Merge new stats with existing
- Line 32-35: Update display if interval elapsed

**Display Stats: Lines 38-74** ← THREAD SAFETY ISSUE
- Line 74: Print statement without lock protection
- Line 75: Flush to ensure output

---

## 7. CONFIGURATION

### maveric/config.py

**MAVERICConfig Dataclass: Lines 11-200**

**Critical Config Fields:**
- Line 20: `clip_model: str = "ViT-B/32"`
- Line 21: `device: str = "auto"`
- Line 24: `cache_base_dir: str = "./cache"`
- Line 25: `enable_image_cache: bool = True`
- Line 31: `request_timeout: int = 5` (Not used in cache_manager!)
- Line 33: `retrieval_rotation_size: int = 1000`
- Line 34: `max_retries: int = 3`
- Line 38-45: `quality_metrics` list (organized by category)
- Line 47-53: `metric_weights` for composite scoring
- Line 57-70: `default_thresholds` (organized by metric)
- Line 89: **`enable_target_class_quality: bool = True`** ← DEFAULT ENABLES EXPENSIVE EFFICIENTNET

---

## 8. DATA FLOW AND INTERFACES

### maveric/core/interfaces.py

**RetrievalResult Dataclass: Lines 14-256**

**Key Methods:**
- Line 37-42: `__post_init__()` - Calculate statistics
- Line 77-80: `to_dataframe()` - Convert to pandas DataFrame
- Line 82-100: `save()` - Save to JSON
- Line 102-123: `load()` - Load from JSON
- Line 203-255: `from_rotation_files()` - Load from rotation file directory
  - Line 226: Pattern for rotation files: `{dataset_name.lower()}_raw_maveric_dataset*.json`
  - Line 227: `sorted(input_path.glob(pattern))` - Sort files
  - Line 234-241: Load each rotation file and extend samples

---

## 9. DATASET HANDLERS

### maveric/retrieval/dataset_handlers.py

**REACTDatasetHandler Class: Lines 30-70**

**Initialization: Lines 33-43**
- Line 49: `load_dataset()` from HuggingFace

**Dataset Loading: Lines 45-52**
- Line 49: `load_dataset(self.dataset_name, split='train')`
- Line 50: Log dataset size

**Iteration: Lines 54-61**
- Line 56: Iterate over self._dataset
- Line 57-60: Return dict with 'URL' and 'TEXT' keys

**Skip Implementation: Lines 67-70**
- Line 69: `self._dataset = self._dataset.skip(n)`
- Line 70: Return self

---

## 10. SUMMARY TABLE: CRITICAL CODE LOCATIONS

| Issue | File | Line(s) | Severity |
|-------|------|---------|----------|
| **O(n×c) Reference Selection** | elevater_datasets.py | 493-530 | CRITICAL |
| **5s Network Timeout** | cache_manager.py | 246 | CRITICAL |
| **Unprotected File Write** | retriever.py | 234-235 | CRITICAL |
| **Weak Cache Validation** | retriever.py | 118-133 | HIGH |
| **EfficientNet Enabled Default** | config.py | 89 | HIGH |
| **No Model Load Progress** | retriever.py | 66,78-85 | MEDIUM |
| **Thread Unsafe Console Output** | progress.py | 74 | MEDIUM |
| **No Connection Pooling** | cache_manager.py | 246 | MEDIUM |

---

## 11. RELATED GIT COMMITS

```
b4fe424 - Log added for blocked retrieval process debugging
f8313c8 - Performance improvement at data retrieval process  (shows O(n) fix)
8394e83 - Data retrival process stuck following logs added
```

Commit f8313c8 contains the solution for the O(n×c) complexity issue.

