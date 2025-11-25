# Potential Bottlenecks Checklist

Based on your observation of 10-20 seconds per sample regardless of cache hits or CPU/GPU, here are all potential causes:

## ✅ FIXED Issues:
1. **sklearn import inside loop** - FIXED (moved to module level)
2. **max_retries/request_timeout not passed** - FIXED (now reads from config)

## 🔍 TO INVESTIGATE (Use debug_retrieval_timing.py):

### 1. **Cache Read Performance** (Suspect!)
**Symptom**: Even cache hits take 10-20 seconds
**Cause**: Google Drive NFS is SLOW for random file access
**Check**: Look at `1_cache_check` and `2_cache_data_extraction` timings
**Solution**: If cache reads are slow (>1s each):
```python
# Consider disabling cache or using local disk
enable_sample_cache: false  # in config
```

### 2. **JSON Parsing Overhead** (High Suspect!)
**Symptom**: Cache hits still slow
**Cause**: Each cache file is ~17KB with base64-encoded embeddings
**Issue**: JSON parsing base64 strings is SLOW
**Check**: If `1_cache_check` > 1 second
**Solution**: Switch cache format from JSON to pickle or numpy:
```python
# In sample_cache_manager.py, use pickle instead of JSON
import pickle
with open(cache_path, 'wb') as f:
    pickle.dump(data, f)
```

### 3. **Per-Class Similarity Loop** (Medium Suspect)
**Symptom**: Time scales with number of classes
**Cause**: 4 cosine_similarity calls per class
**Check**: If `8_per_class_scores` > 1 second for 10 classes
**For CIFAR-10 (10 classes)**: Should be < 0.1s
**For CIFAR-100 (100 classes)**: Could be 1-2s
**Solution**: Batch the cosine_similarity calls:
```python
# Instead of 4 calls per class, do 1 call for all classes
img2img_all = cosine_similarity(img_embedding, all_ref_embeddings)
```

### 4. **Google Drive Sync Delays** (High Suspect!)
**Symptom**: Random pauses, inconsistent timing
**Cause**: Google Drive syncing files in background
**Check**: Run `watch -n 1 "ps aux | grep drive"` to see if Drive is busy
**Solution**:
```bash
# Stop Google Drive sync during retrieval
pkill drive  # or pause sync in Drive settings
```

### 5. **Sample Cache File Fragmentation**
**Symptom**: Many small files (270K+ cache files)
**Cause**: Google Drive is slow with many small files
**Check**: `ls -la /path/to/sample_metadata_cache | wc -l`
**Solution**: Batch cache into larger files (100 samples per file)

### 6. **Language Detection in Semantic Metrics** (Possible)
**Symptom**: Time in `5_semantic_metrics` > 0.5s
**Cause**: `langdetect` library can be slow
**Check**: If semantic metrics take > 0.5s
**Solution**: Disable text quality metric or cache language detection

### 7. **Base64 Encoding/Decoding** (Possible)
**Symptom**: Time in `7_cache_save` or `2_cache_data_extraction` > 0.5s
**Cause**: Encoding/decoding 16KB embeddings to/from base64
**Check**: If cache operations > 0.5s
**Solution**: Store embeddings as binary pickle instead of JSON+base64

### 8. **Reference Embeddings Not Cached** (Low Probability)
**Symptom**: First sample after restart is very slow
**Cause**: Loading reference embeddings on every run
**Check**: Look for "Preparing reference embeddings" log
**Solution**: Already cached, but verify cache is being used

## 🎯 Most Likely Culprits (in order):

1. **Google Drive file I/O** - Every cache read/write hits slow NFS
2. **JSON+base64 overhead** - Parsing 17KB JSON files repeatedly
3. **Multiple small file operations** - 270K cache files overwhelm Google Drive

## Quick Test to Isolate:

### Test 1: Disable Cache Completely
```yaml
# In maveric_config.yaml
enable_sample_cache: false
```
**If retrieval speeds up**: Cache I/O is the bottleneck
**If still slow**: Something else is the issue

### Test 2: Use Local Disk Instead of Google Drive
```bash
# Copy cache to local disk
cp -r /content/drive/MyDrive/MAVERIC/maveric_cache /tmp/maveric_cache

# Update config
cache_base_dir: "/tmp/maveric_cache"
```
**If retrieval speeds up**: Google Drive NFS is the bottleneck

### Test 3: Single Sample Timing
```python
# Run just ONE sample and time each step manually
import time
start = time.time()
# ... your retrieval code
print(f"Time: {time.time() - start}")
```

## Action Plan:

1. **IMMEDIATE**: Run `debug_retrieval_timing.py` and send me the output
2. **QUICK FIX**: Try Test 1 (disable cache) to isolate
3. **BEST FIX**: Switch from Google Drive to local SSD for cache
4. **ALTERNATIVE**: Change cache format from JSON to pickle

## Expected Normal Timings:

```
Cache HIT:
  - Total: 0.1-0.3s
  - Most time in: per_class_scores (0.05-0.1s)

Cache MISS:
  - Total: 1-3s
  - Most time in: download (0.5-2s) + embeddings (0.1-0.3s)
```

If you're seeing 10-20 seconds, one of the above is the culprit!
