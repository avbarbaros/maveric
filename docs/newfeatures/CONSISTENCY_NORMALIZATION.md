# Consistency Score Normalization and Null-Model Testing

**Date**: July 23, 2026  
**Purpose**: Address reviewer concern about scale normalization and mechanical correlation

## Reviewer Concern

> "The consistency score is defined as 1 minus the standard deviation of four directional similarity metrics. This assumes metrics share the same scale/distribution. As a result, the score may mix modality-specific similarities. Without normalization, null-model tests, or controls for boundedness, the paper cannot establish that this correlation reflects real multimodal quality structure rather than an artifact of the scoring design."

## Implementation

### 1. Z-Score Normalization

**Formula**:
```python
def consistency_score(metrics, normalization="none", eps=1e-8):
    # metrics: (N, 4) array for ONE class: [q_i2i, q_t2t, q_i2t, q_t2i]
    if normalization == "zscore":
        mu, sd = metrics.mean(0), metrics.std(0)
        metrics = (metrics - mu) / (sd + eps)   # per-class, per-metric z-score
    return 1.0 - metrics.std(axis=1)            # q_avg stays on RAW metrics
```

**Key properties**:
- **Per-class normalization**: Each class has its own μ and σ for each metric
- **Per-metric normalization**: Each of the 4 metrics normalized independently
- **Weighted average unchanged**: `q_avg` still computed on raw similarities
- **Only consistency affected**: Z-scoring only applies to consistency calculation

### 2. Null-Model Permutation Test

**Purpose**: Test whether correlation between `weighted_class_score` and `consistency` is mechanical or reflects real quality structure.

**Algorithm**:
```python
def null_test(metrics, B=1000, rng=np.random.default_rng(0)):
    q_avg  = metrics.mean(1)
    q_cons = 1.0 - metrics.std(1)
    rho_obs = np.corrcoef(q_avg, q_cons)[0, 1]
    
    rho_null = np.empty(B)
    for b in range(B):
        perm = np.column_stack([rng.permutation(metrics[:, j]) for j in range(4)])
        rho_null[b] = np.corrcoef(perm.mean(1), 1.0 - perm.std(1))[0, 1]
    
    return rho_obs, rho_null.mean(), rho_null.std(), np.quantile(rho_null, [.025, .975])
```

**Interpretation**:
- **p < 0.05**: Correlation significantly different from null (real quality structure)
- **p > 0.05**: Correlation may be mechanical artifact
- **Many significant classes**: Evidence for real multimodal quality
- **Few significant classes**: Consider enabling z-score normalization

## Configuration

### Enable Z-Score Normalization

**File**: `experiments/maveric_config.yaml`

```yaml
# Consistency score normalization configuration
consistency_normalization: "zscore"  # Options: "none" (default) or "zscore"
```

**Effect**:
- During data retrieval, raw similarities are stored as usual
- Consistency scores are computed with per-class, per-metric z-score normalization
- Weighted average (`weighted_class_score`) remains unchanged

## Usage

### 1. Run Null-Model Analysis (Recommended First Step)

Test whether normalization is needed:

```bash
python -m maveric.utils.consistency_analysis \
    --data results/cifar10/curated/training_data.json \
    --normalization none \
    --iterations 1000 \
    --output results/cifar10/analysis/null_test_report.txt
```

**Output example**:
```
================================================================================
CONSISTENCY SCORE NULL-MODEL ANALYSIS
================================================================================

Total classes analyzed: 10
Classes with significant correlation (p < 0.05): 8 (80.0%)

PER-CLASS RESULTS:

Class                     N   ρ_obs  ρ_null p-value   Sig
--------------------------------------------------------------------------------
airplane                450   0.234   0.012   0.001   ***
automobile              450   0.189  -0.008   0.003    **
bird                    450   0.301   0.015   0.000   ***
...
--------------------------------------------------------------------------------

Significance codes: *** p<0.001, ** p<0.01, * p<0.05

================================================================================
INTERPRETATION:
================================================================================
✅ STRONG EVIDENCE for real multimodal quality structure
   Most classes show significant correlation beyond mechanical effects.
```

### 2. Compare with Z-Score Normalization

Test if normalization changes results:

```bash
python -m maveric.utils.consistency_analysis \
    --data results/cifar10/curated/training_data.json \
    --normalization zscore \
    --iterations 1000 \
    --output results/cifar10/analysis/null_test_zscore_report.txt
```

### 3. Apply Z-Score Normalization to Data (Post-Processing)

If null-test shows mechanical correlation, apply z-score normalization:

```bash
python -m maveric.utils.consistency_analysis \
    --data results/cifar10/curated/training_data.json \
    --apply-zscore results/cifar10/curated/training_data_zscore.json
```

This creates a new dataset with z-score normalized consistency scores.

### 4. Enable During Data Retrieval

For future retrievals, enable in config:

```yaml
# maveric_config.yaml
consistency_normalization: "zscore"
```

Then run retrieval as normal:

```bash
python experiments/01_data_retrieval.py --config maveric_config.yaml --dataset cifar10
```

## API Reference

### `consistency_analysis.py` Functions

#### `compute_consistency_score()`
```python
def compute_consistency_score(metrics: np.ndarray,
                              normalization: str = "none",
                              eps: float = 1e-8) -> np.ndarray:
    """
    Compute consistency scores with optional normalization.

    Args:
        metrics: Shape (N, 4) - [img2img, txt2txt, img2txt, txt2img] for ONE class
        normalization: "none" or "zscore"
        eps: Small constant to avoid division by zero

    Returns:
        Consistency scores (N,)
    """
```

#### `null_test_correlation()`
```python
def null_test_correlation(metrics: np.ndarray,
                          B: int = 1000,
                          seed: int = 0,
                          normalization: str = "none") -> Dict:
    """
    Permutation test for correlation between weighted_score and consistency.

    Args:
        metrics: Shape (N, 4) - similarity metrics
        B: Number of permutation iterations
        seed: Random seed
        normalization: Normalization method

    Returns:
        Dict with:
            - rho_observed: Observed correlation
            - rho_null_mean: Mean of null distribution
            - rho_null_std: Std of null distribution
            - ci_95: 95% confidence interval
            - p_value: Two-tailed p-value
            - significant: Whether p < 0.05
    """
```

#### `analyze_curated_data()`
```python
def analyze_curated_data(data_path: str,
                        class_column: str = 'label',
                        normalization: str = "none",
                        B: int = 1000,
                        seed: int = 0) -> Dict[str, Dict]:
    """
    Run null-test analysis for all classes in dataset.

    Returns:
        Dict mapping class_name -> null_test_results
    """
```

#### `apply_zscore_normalization_to_data()`
```python
def apply_zscore_normalization_to_data(data_path: str,
                                       output_path: str,
                                       class_column: str = 'label') -> None:
    """
    Apply z-score normalization to consistency scores and save.

    Updates all Class_{name}_consistency columns with z-score normalized values.
    """
```

## Files Modified

1. **`maveric/config.py`** (line 73-75)
   - Added `consistency_normalization` config parameter

2. **`maveric/retrieval/retriever.py`** (lines 38-110, 732-745)
   - Added `consistency_normalization` parameter to `__init__`
   - Added buffer to store similarities for potential normalization
   - Updated consistency calculation with normalization support

3. **`maveric/main.py`** (line 88)
   - Pass `consistency_normalization` from config to Retriever

4. **`maveric/quality/metrics/multimodal_metrics.py`** (lines 23-56, 107-136)
   - Added `normalization` parameter to `MultimodalConsistencyMetric`
   - Implemented z-score normalization in `compute()` method

5. **`experiments/maveric_config.yaml`** (lines 39-68)
   - Added detailed documentation and configuration option

6. **`maveric/utils/consistency_analysis.py`** (NEW FILE)
   - Complete analysis toolkit with null-testing and normalization

7. **`docs/newfeatures/CONSISTENCY_NORMALIZATION.md`** (NEW FILE - this file)
   - Comprehensive documentation

## Expected Results

### Without Normalization (Default)
- **Use case**: Metrics have similar scales/distributions
- **Behavior**: Original MAVERIC consistency calculation
- **When to use**: Null-test shows significant correlation (p < 0.05) for most classes

### With Z-Score Normalization
- **Use case**: Metrics have different scales/distributions
- **Behavior**: Per-class, per-metric standardization before consistency calculation
- **When to use**: Null-test shows weak/mechanical correlation
- **Expected effect**: More classes show significant correlation beyond mechanical effects

## Workflow Summary

1. **Retrieve data** with default settings (`consistency_normalization: "none"`)
2. **Run null-test** to assess mechanical vs. real correlation
3. **Decide**:
   - If most classes significant (p < 0.05): ✅ Use as-is
   - If few classes significant: ⚠️ Enable z-score normalization
4. **Optional**: Apply z-score post-processing to existing data
5. **Future retrievals**: Enable in config if needed

## References

- Reviewer comment on consistency score design
- Statistical permutation testing methodology
- Z-score normalization for heterogeneous metric scales

## Status

✅ **IMPLEMENTED** - Ready for use (July 23, 2026)
