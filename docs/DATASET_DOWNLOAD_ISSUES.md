# Dataset Download Issues and Solutions

**Date**: November 5, 2025
**Issue**: Broken torchvision download URLs for academic datasets

---

## Problem

Some ELEVATER datasets hosted by torchvision have **broken download URLs** due to:
- Academic servers being taken offline
- URL changes without torchvision updates
- Domain expiration
- Server maintenance

**Common Errors**:
- `HTTP Error 404: Not Found`
- `URLError: <urlopen error [Errno -2] Name or service not known>`
- Connection timeouts

---

## Affected Datasets

| Dataset | Status | Download URL Status | Solution |
|---------|--------|---------------------|----------|
| **Caltech101** | ⚠️ Broken | 404 Not Found | Manual download |
| **Country211** | ⚠️ May Fail | Unstable | Use alternative |
| **CIFAR-10** | ✅ Working | Stable | Auto-download OK |
| **CIFAR-100** | ✅ Working | Stable | Auto-download OK |
| **Food101** | ✅ Working | Stable | Auto-download OK (5GB) |
| **MNIST** | ✅ Working | Stable | Auto-download OK |
| **Oxford Flowers102** | ✅ Working | Stable | Auto-download OK |
| **Oxford Pets** | ✅ Working | Stable | Auto-download OK |
| **GTSRB** | ✅ Working | Stable | Auto-download OK |
| **EuroSAT** | ✅ Working | Stable | Auto-download OK |

---

## Solutions

### Option 1: Use Working Datasets (Recommended)

**Quick Start - Use Reliable Datasets**:
```bash
# These datasets work perfectly with auto-download:
python experiments/01_data_retrieval.py --config config.yaml --dataset cifar10
python experiments/01_data_retrieval.py --config config.yaml --dataset cifar100
python experiments/01_data_retrieval.py --config config.yaml --dataset food101
python experiments/01_data_retrieval.py --config config.yaml --dataset mnist
```

**Recommended Dataset Order**:
1. **CIFAR-10** - Small (170MB), 10 classes, fast download
2. **CIFAR-100** - Small (170MB), 100 classes, fast download
3. **MNIST** - Tiny (11MB), 10 classes, fastest
4. **Oxford Pets** - Medium (800MB), 37 classes
5. **Food101** - Large (5GB), 101 classes, comprehensive

---

### Option 2: Manual Download for Broken Datasets

#### Caltech101 Manual Download

**Step 1: Download from official source**:
```bash
# Download dataset (125MB)
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz

# Or use curl
curl -O http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
```

**Step 2: Extract to correct location**:
```bash
# Extract to MAVERIC cache directory
tar -xzf 101_ObjectCategories.tar.gz -C /content/drive/MyDrive/MAVERIC/maveric_cache/datasets/caltech101/

# Verify structure
ls /content/drive/MyDrive/MAVERIC/maveric_cache/datasets/caltech101/
# Should show: 101_ObjectCategories/ directory
```

**Step 3: Run retrieval**:
```bash
# MAVERIC will detect existing data and skip download
python experiments/01_data_retrieval.py --config config.yaml --dataset caltech101
```

#### Alternative: Use Kaggle Datasets

Many academic datasets are mirrored on Kaggle:

```bash
# Install Kaggle CLI
pip install kaggle

# Download from Kaggle (requires API token)
kaggle datasets download -d jessicali9530/caltech101
unzip caltech101.zip -d /content/drive/MyDrive/MAVERIC/maveric_cache/datasets/caltech101/
```

---

### Option 3: Skip Problematic Datasets

**For Multi-Dataset Experiments**:

Create a dataset list excluding broken ones:

```python
# experiments/batch_retrieval_safe.py
working_datasets = [
    'cifar10', 'cifar100', 'mnist',
    'food101', 'oxford_pets', 'oxford_flowers102',
    'gtsrb', 'eurosat'
]

for dataset in working_datasets:
    print(f"\n{'='*60}")
    print(f"Processing: {dataset}")
    print('='*60)

    result = maveric.retrieve(
        dataset_name="react-vl/react-retrieval-datasets",
        target_dataset=dataset,
        num_samples=1000
    )
```

---

## Error Messages and Fixes

### Error 1: HTTP 404 Not Found

**Error Message**:
```
urllib.error.HTTPError: HTTP Error 404: Not Found
Failed to load caltech101 dataset: HTTP Error 404: Not Found
```

**Solution**:
```bash
# Option A: Manual download (see above)
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz

# Option B: Use different dataset
python experiments/01_data_retrieval.py --config config.yaml --dataset cifar10
```

### Error 2: Connection Timeout

**Error Message**:
```
URLError: <urlopen error timed out>
```

**Solution**:
```bash
# 1. Check internet connection
ping google.com

# 2. Increase timeout in config
# Edit maveric_config.yaml:
request_timeout: 30  # Increase from 15 to 30 seconds

# 3. Try different network or VPN
```

### Error 3: Name Resolution Error

**Error Message**:
```
URLError: <urlopen error [Errno -2] Name or service not known>
```

**Solution**:
```bash
# DNS issue - try:
# 1. Wait and retry (server may be down temporarily)
# 2. Check /etc/resolv.conf for DNS settings
# 3. Use different dataset
```

---

## Automated Fix (Implemented)

**Version**: November 5, 2025
**Location**: `maveric/datasets/elevater_datasets.py`

**Enhanced Error Handling**:
```python
try:
    self._dataset = dataset_loaders[self.dataset_name]()
except urllib.error.HTTPError as e:
    if e.code == 404:
        error_msg = (
            f"Failed to download {self.dataset_name}: URL is broken (404).\n\n"
            f"📥 Manual Download Options:\n"
            f"   1. Download from: [alternative URL]\n"
            f"   2. Extract to: {self.data_dir}/\n"
            f"   3. Use different dataset (cifar10, cifar100, food101)\n"
        )
        raise DatasetError(error_msg)
```

**What This Does**:
- ✅ Catches HTTP 404 errors specifically
- ✅ Provides helpful manual download instructions
- ✅ Suggests working alternative datasets
- ✅ Shows exact extraction path
- ✅ Gives clear next steps

---

## Dataset Characteristics

### Recommended Datasets for Testing

**Small & Fast** (Good for testing):
| Dataset | Size | Classes | Samples | Download Time |
|---------|------|---------|---------|---------------|
| MNIST | 11MB | 10 | 60,000 | <1 min |
| CIFAR-10 | 170MB | 10 | 50,000 | ~2 min |
| CIFAR-100 | 170MB | 100 | 50,000 | ~2 min |
| EuroSAT | 90MB | 10 | 27,000 | ~1 min |

**Medium** (Good for experiments):
| Dataset | Size | Classes | Samples | Download Time |
|---------|------|---------|---------|---------------|
| Oxford Pets | 800MB | 37 | 7,349 | ~5 min |
| Oxford Flowers102 | 350MB | 102 | 8,189 | ~3 min |
| GTSRB | 300MB | 43 | 39,209 | ~3 min |

**Large** (Production use):
| Dataset | Size | Classes | Samples | Download Time |
|---------|------|---------|---------|---------------|
| Food101 | 5GB | 101 | 101,000 | ~15-30 min |
| Caltech101 | 125MB | 102 | 9,146 | Manual only |

---

## Prevention Tips

### 1. Check Dataset Before Large-Scale Retrieval

```python
# Test with small sample first
from maveric import MAVERIC

maveric = MAVERIC.from_config_file('config.yaml')

try:
    # Test with just 10 samples
    result = maveric.retrieve(
        dataset_name="react-vl/react-retrieval-datasets",
        target_dataset="caltech101",
        num_samples=10
    )
    print("✅ Dataset works! Proceeding with full retrieval...")
except Exception as e:
    print(f"❌ Dataset failed: {e}")
    print("Switching to alternative dataset...")
    # Use working dataset instead
```

### 2. Monitor Download Progress

```bash
# Watch download directory size
watch -n 1 du -sh /content/drive/MyDrive/MAVERIC/maveric_cache/datasets/

# Check network activity
iftop  # Or similar tool
```

### 3. Use Cached Data for Sample Caching

```yaml
# config.yaml
enable_sample_cache: true  # Once first dataset works, subsequent ones are faster
```

---

## Troubleshooting Checklist

When you encounter dataset download issues:

- [ ] Check if dataset is in "Affected Datasets" list above
- [ ] Try a known-working dataset first (CIFAR-10, MNIST)
- [ ] Check internet connection and firewall settings
- [ ] Look for alternative download sources (Kaggle, official site)
- [ ] Consider manual download if auto-download fails
- [ ] Check available disk space (Food101 needs 5GB+)
- [ ] Increase `request_timeout` in config if experiencing timeouts
- [ ] Check MAVERIC GitHub issues for similar problems
- [ ] Use file-based datasets (DTD, FER2013) as alternative

---

## Quick Reference Commands

**Test Connection**:
```bash
# Test if dataset URL is accessible
curl -I http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
```

**Check Dataset Directory**:
```bash
# See what datasets are already downloaded
ls -lh /content/drive/MyDrive/MAVERIC/maveric_cache/datasets/
```

**Clear Failed Downloads**:
```bash
# Remove incomplete downloads
rm -rf /content/drive/MyDrive/MAVERIC/maveric_cache/datasets/caltech101/
```

**Verify Dataset Structure**:
```python
# Check if dataset loads
from maveric.datasets import get_dataset
dataset = get_dataset('cifar10', root='/path/to/cache/datasets')
print(f"Dataset loaded: {len(dataset)} samples")
```

---

## Summary

**Best Practice Workflow**:
1. ✅ Start with working datasets (CIFAR-10, CIFAR-100)
2. ✅ Use sample caching to benefit from first retrieval
3. ✅ Test with small sample counts first (10-100 samples)
4. ✅ Manually download problematic datasets when needed
5. ✅ Keep dataset cache on fast, reliable storage

**Avoid**:
- ❌ Don't start with Caltech101 or Country211
- ❌ Don't use Google Drive for large downloads (use local storage first, then copy)
- ❌ Don't ignore download errors - fix before proceeding

---

**Document Version**: 1.0
**Last Updated**: November 5, 2025
**Status**: Caltech101 confirmed broken, fix implemented
