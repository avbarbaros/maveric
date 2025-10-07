# MAVERIC Hyperparameter Search Guide

Systematic hyperparameter tuning to optimize model performance beyond the current baseline.

## Quick Start

### Based on Your Results

Your current experiments show:
- `regularization_weight=0.50`: **92.14%** ✓ (best)
- `regularization_weight=0.25`: 91.85%
- `regularization_weight=0.10`: 91.20%
- `regularization_weight=0.75`: 91.37%

**Goal**: Reach 92.50% accuracy

### Recommended Search Strategy

Since `regularization_weight=0.5` appears optimal, search other parameters:

```bash
# 1. Focused search (recommended first step)
python 05_hyperparameter_search.py \
    --input /path/to/your/training_data/ \
    --config maveric_config.yaml \
    --output ./hp_search_results/focused/ \
    --search-type focused

# 2. Learning rate optimization (if focused search doesn't reach goal)
python 05_hyperparameter_search.py \
    --input /path/to/your/training_data/ \
    --config maveric_config.yaml \
    --output ./hp_search_results/lr_search/ \
    --search-type learning_rate
```

## Search Types

### 1. Focused Search (Recommended)
Searches around optimal `regularization_weight=0.5` while varying complementary parameters:
- `regularization_weight`: [0.40, 0.45, 0.50, 0.55, 0.60]
- `learning_rate`: [5e-7, 1e-6, 1.5e-6, 2e-6]
- `weight_decay`: [0.005, 0.01, 0.015]
- `epochs`: [10, 15]

**Grid size**: ~120 experiments

```bash
python 05_hyperparameter_search.py \
    -i data/training/ \
    -c maveric_config.yaml \
    -o results/focused/ \
    --search-type focused
```

### 2. Regularization Search
Fine-grained search only on `regularization_weight`:
- `regularization_weight`: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
- Other parameters: fixed at current values

**Grid size**: 7 experiments

```bash
python 05_hyperparameter_search.py \
    -i data/training/ \
    -c maveric_config.yaml \
    -o results/reg_search/ \
    --search-type regularization
```

### 3. Learning Rate Search
Optimize learning rate with fixed optimal regularization:
- `regularization_weight`: [0.5] (fixed)
- `learning_rate`: [1e-7, 5e-7, 8e-7, 1e-6, 1.2e-6, 1.5e-6, 2e-6, 3e-6]
- `weight_decay`: [0.005, 0.01, 0.015]

**Grid size**: ~24 experiments

```bash
python 05_hyperparameter_search.py \
    -i data/training/ \
    -c maveric_config.yaml \
    -o results/lr_search/ \
    --search-type learning_rate
```

### 4. Broad Search
Comprehensive multi-dimensional search (expensive):
- All training hyperparameters
- Optimizers, schedulers, warmup steps

**Grid size**: ~1000+ experiments

```bash
python 05_hyperparameter_search.py \
    -i data/training/ \
    -c maveric_config.yaml \
    -o results/broad/ \
    --search-type broad \
    --method random \
    -n 50  # Random sampling recommended
```

## Search Methods

### Grid Search (Default)
Exhaustive search over all parameter combinations:
```bash
python 05_hyperparameter_search.py \
    -i data/training/ \
    -c maveric_config.yaml \
    -o results/grid/ \
    --method grid
```

### Random Search
Random sampling from search space (faster):
```bash
python 05_hyperparameter_search.py \
    -i data/training/ \
    -c maveric_config.yaml \
    -o results/random/ \
    --method random \
    -n 20  # Number of random samples
```

## Resuming Interrupted Searches

If a search is interrupted, resume from the last completed experiment:

```bash
python 05_hyperparameter_search.py \
    -i data/training/ \
    -c maveric_config.yaml \
    -o results/focused/ \
    --search-type focused \
    --resume-from 15  # Resume from experiment 15
```

## Understanding Results

### Output Structure

```
hp_search_results/
├── search_summary.json          # Summary of all experiments
├── exp_001/
│   ├── config.yaml             # Configuration used
│   ├── result.json             # Experiment results
│   └── models/
│       └── best_model.pth      # Best checkpoint
├── exp_002/
│   ├── config.yaml
│   ├── result.json
│   └── models/
└── ...
```

### search_summary.json

```json
{
  "timestamp": "2025-01-07T10:30:00",
  "total_experiments": 120,
  "best_result": {
    "experiment_id": 47,
    "parameters": {
      "regularization_weight": 0.55,
      "learning_rate": 1.5e-06,
      "weight_decay": 0.005,
      "epochs": 15
    },
    "metrics": {
      "test_accuracy": 92.67,
      "zero_shot_baseline": 89.34,
      "improvement": 3.33
    }
  },
  "all_results": [...]
}
```

### Analyzing Results

The script automatically prints:

1. **Top 5 Configurations** - Best performing parameter sets
2. **Best Configuration Details** - Optimal hyperparameters
3. **Parameter Impact Analysis** - Average accuracy per parameter value

Example output:
```
🏆 Top 5 Configurations:

1. Experiment 47: 92.67%
   Parameters:
     regularization_weight: 0.55
     learning_rate: 1.5e-06
     weight_decay: 0.005
     epochs: 15

2. Experiment 23: 92.51%
   Parameters:
     regularization_weight: 0.50
     learning_rate: 2e-06
     weight_decay: 0.01
     epochs: 15

...

PARAMETER IMPACT ANALYSIS

regularization_weight:
  0.55: 92.45% ± 0.15% (n=24)
  0.50: 92.21% ± 0.18% (n=24)
  0.60: 91.98% ± 0.22% (n=24)
  0.45: 91.87% ± 0.19% (n=24)
  0.40: 91.72% ± 0.21% (n=24)

learning_rate:
  1.5e-06: 92.34% ± 0.23% (n=30)
  2e-06: 92.18% ± 0.26% (n=30)
  1e-06: 91.95% ± 0.21% (n=30)
  5e-07: 91.76% ± 0.19% (n=30)
```

## Recommended Workflow to Reach 92.50%

### Step 1: Run Focused Search
```bash
python 05_hyperparameter_search.py \
    -i /path/to/training_data/ \
    -c maveric_config.yaml \
    -o results/step1_focused/ \
    --search-type focused
```

**Expected time**: ~1-2 hours per experiment × 120 = ~5-10 days
**Cost**: Moderate (120 full training runs)

### Step 2: Analyze and Refine

Check `results/step1_focused/search_summary.json`:
- If best accuracy ≥ 92.50%: **Done!** Use those parameters
- If best accuracy < 92.50%: Identify promising parameter ranges

### Step 3: Run Learning Rate Search (if needed)
```bash
python 05_hyperparameter_search.py \
    -i /path/to/training_data/ \
    -c maveric_config.yaml \
    -o results/step2_lr/ \
    --search-type learning_rate
```

### Step 4: Fine-tune Top Configuration

Manually adjust the best configuration found:
- Slightly increase `epochs` (15 → 20)
- Try adjacent `learning_rate` values
- Adjust `warmup_steps` (0 → 50 → 100)

## Performance Tips

### Parallel Execution

Run multiple experiments in parallel on different GPUs:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python 05_hyperparameter_search.py \
    -i data/training/ -c config.yaml -o results/gpu0/ \
    --search-type focused --resume-from 0

# GPU 1 (in separate terminal)
CUDA_VISIBLE_DEVICES=1 python 05_hyperparameter_search.py \
    -i data/training/ -c config.yaml -o results/gpu1/ \
    --search-type focused --resume-from 30
```

### Time Estimates

Based on typical CLIP fine-tuning:
- Single experiment (10 epochs): ~1-2 hours on T4 GPU
- Focused search (120 configs): ~5-10 days (sequential)
- Regularization search (7 configs): ~7-14 hours
- Learning rate search (24 configs): ~1-2 days

### Cost Optimization

**Start small, scale up:**
1. Run `regularization` search first (7 experiments)
2. If promising, run `learning_rate` search (24 experiments)
3. Only run `focused` if gap remains (120 experiments)

**Use random search for broad exploration:**
```bash
# Sample 30 random configurations from broad space
python 05_hyperparameter_search.py \
    -i data/training/ -c config.yaml -o results/random/ \
    --search-type broad --method random -n 30
```

## Common Issues

### Out of Memory
Reduce `batch_size` in `maveric_config.yaml`:
```yaml
batch_size: 16  # Instead of 32
```

### Slow Training
- Enable mixed precision: `mixed_precision: true`
- Reduce `epochs`: Use 10 instead of 15 for initial search
- Use `num_workers: 4` for faster data loading

### Poor Results
- Check training data quality thresholds
- Verify class balance in training data
- Ensure test set matches target distribution

## Custom Search Space

Edit `05_hyperparameter_search.py` to define custom search spaces:

```python
def define_search_space(self, search_type: str = "custom"):
    if search_type == "custom":
        return {
            'regularization_weight': [0.48, 0.50, 0.52],  # Fine-grained around best
            'learning_rate': [1.2e-6, 1.5e-6, 1.8e-6],
            'weight_decay': [0.008, 0.01, 0.012],
            'epochs': [12, 15, 18],
            'warmup_steps': [0, 50, 100]
        }
```

Then run:
```bash
python 05_hyperparameter_search.py \
    -i data/training/ -c config.yaml -o results/custom/ \
    --search-type custom
```

## Best Practices

1. **Start with small searches** - Validate the setup works before large searches
2. **Monitor intermediate results** - Check `search_summary.json` periodically
3. **Save checkpoints** - Always enable `save_best_model: true`
4. **Document everything** - Each experiment directory contains full config
5. **Use version control** - Track which search led to best results

## Next Steps After Finding Optimal Parameters

1. **Validate on multiple runs** - Ensure results are reproducible
2. **Update base config** - Set optimal parameters as defaults
3. **Document findings** - Record parameter values and accuracy gains
4. **Test generalization** - Validate on other datasets if applicable
