# Retrieval Timing Debug Instructions

## How to Use the Timing Instrumentation

### Step 1: Import the debug script at the top of 01_data_retrieval.py

Add this line **after the imports but BEFORE creating MAVERIC**:

```python
# At the top of experiments/01_data_retrieval.py, add:
import sys
sys.path.insert(0, '/workspaces/maveric')  # or your maveric path
import debug_retrieval_timing  # This patches the Retriever with timing
```

### Step 2: Run your retrieval script normally

```bash
python experiments/01_data_retrieval.py --config maveric_config.yaml
```

### Step 3: Observe the detailed timing output

You'll see output like this for EACH sample:

```
================================================================================
🔍 TIMING ANALYSIS FOR SAMPLE: https://example.com/image.jpg...
================================================================================
⏱️  Cache check: 0.0012s (MISS)
⏱️  Image download: 2.3456s
⏱️  CLIP embeddings: 0.0234s
⏱️  Visual metrics: 0.0156s
⏱️  Semantic metrics: 0.0089s
⏱️  EfficientNet: 0.0000s
⏱️  Cache save: 0.0234s
📊 Computing scores for 10 classes...
⏱️  Per-class similarity (10 classes): 0.1234s
   └─ Per-class average: 0.0123s

================================================================================
📊 TOTAL TIME: 2.5415s
================================================================================

⚠️  BREAKDOWN:
   1_cache_check                 :   0.0012s (  0.0%)
   2_download                    :   2.3456s ( 92.3%) ⚠️ BOTTLENECK!
   3_clip_embeddings             :   0.0234s (  0.9%)
   4_visual_metrics              :   0.0156s (  0.6%)
   5_semantic_metrics            :   0.0089s (  0.4%)
   6_efficientnet                :   0.0000s (  0.0%)
   7_cache_save                  :   0.0234s (  0.9%)
   8_per_class_scores            :   0.1234s (  4.9%)
   TOTAL                         :   2.5415s (100.0%)
================================================================================
```

## What to Look For

### Expected Results (Normal):
- **Cache HIT**: Total time should be < 0.5s
  - Most time in `8_per_class_scores` (similarity computation)
  - Almost no time in download/embeddings

- **Cache MISS**: Total time 1-3s
  - Most time in `2_download` (expected)
  - Some time in `3_clip_embeddings` (expected)
  - Small time in metrics and cache save

### Problem Indicators:

1. **If you see 10-20 seconds per sample**:
   - Check which step has high percentage
   - Look for steps taking > 5 seconds

2. **Common bottlenecks**:
   - `2_download` > 5s → Network timeout issues (check `request_timeout`)
   - `8_per_class_scores` > 5s → Too many classes or slow cosine_similarity
   - `7_cache_save` > 2s → Slow disk I/O (Google Drive NFS issue)
   - `1_cache_check` > 1s → Slow cache file reads

3. **Red flags**:
   - ANY step showing > 50% of total time (except download on cache miss)
   - Total time > 5 seconds even on cache HIT
   - `cache_save` taking longer than embedding computation

## Report Back

Copy and paste a few sample outputs showing:
1. One cache HIT sample
2. One cache MISS sample
3. The breakdown percentages

This will immediately show where the 10-20 seconds is being spent!

## Alternative: Quick Single Sample Test

If you want to test just ONE sample without running the full retrieval:

```python
# test_single_sample.py
import sys
sys.path.insert(0, '/workspaces/maveric')
import debug_retrieval_timing

from maveric import MAVERIC
from maveric.config import MAVERICConfig

config = MAVERICConfig(
    cache_base_dir="./cache",
    max_retries=1,
    request_timeout=2
)

maveric = MAVERIC(config)

# Test with a sample URL
result = maveric.retrieve(
    dataset_name="react-vl/react-retrieval-datasets",
    target_dataset="cifar10",
    num_samples=5,  # Just 5 samples for testing
    start_index=0
)
```

This will show timing for just 5 samples, making it easier to analyze.
