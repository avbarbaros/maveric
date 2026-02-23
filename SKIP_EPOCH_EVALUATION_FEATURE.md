# Skip Epoch Evaluation Feature for Unified Training

## Overview

This feature allows disabling per-epoch evaluation during unified training mode, which significantly reduces training time when evaluation on the unified 1,196-class space provides no useful signal.

## Problem Statement

In unified training mode (REACT-style multi-dataset training):
- Model trains on 20 ELEVATER datasets combined (1,196 total classes)
- Each epoch takes ~35-40 minutes
- Per-epoch evaluation on the unified 1,196-class test set:
  - Adds significant time overhead
  - Shows monotonically increasing accuracy (18.28% → 28.55% over 10 epochs)
  - Provides no useful early stopping signal
  - Doesn't reflect real-world per-dataset performance

**Solution**: Skip per-epoch evaluation during training, save periodic checkpoints, and evaluate final model separately on each dataset using `05_unified_evaluation.py`.

## Configuration

### YAML Configuration File

Add to `experiments/maveric_config.yaml`:

```yaml
training:
  # ... other training parameters ...

  # Model checkpointing
  save_best_model: true
  eval_frequency: 1
  save_frequency: 1  # Save checkpoint every N epochs when skip_epoch_evaluation=True
  keep_last_n_checkpoints: 3
  skip_epoch_evaluation: false  # Set to TRUE for unified training
```

### For Unified Training

Set `skip_epoch_evaluation: true` in config:

```yaml
training:
  epochs: 50
  learning_rate: 0.0000001
  skip_epoch_evaluation: true  # ← Enable this
  save_frequency: 5  # Save checkpoint every 5 epochs
```

## Behavior

### When `skip_epoch_evaluation: false` (DEFAULT)

**Standard behavior** for per-dataset training:
- Evaluates on test set every `eval_frequency` epochs
- Saves best model based on test accuracy
- Logs: `Train Loss: X.XXXX, Train Acc: XX.XX%, Test Loss: X.XXXX, Test Acc: XX.XX%`
- **Use this for**: Single-dataset training where test evaluation is meaningful

### When `skip_epoch_evaluation: true` (UNIFIED TRAINING)

**Optimized behavior** for unified training:
- **Skips test evaluation** entirely during training
- Logs only training metrics: `Train Loss: X.XXXX, Train Acc: XX.XX% (evaluation skipped)`
- Saves periodic checkpoints every `save_frequency` epochs
- Checkpoints named: `checkpoint_epoch_1.pth`, `checkpoint_epoch_5.pth`, etc.
- **Use this for**: Unified training where per-dataset evaluation happens separately

## Usage Example

### Step 1: Configure for Unified Training

Edit `experiments/maveric_config.yaml`:

```yaml
training:
  epochs: 60
  learning_rate: 0.0000001
  weight_decay: 0.05
  skip_epoch_evaluation: true  # ← Enable
  save_frequency: 5  # Save every 5 epochs
```

### Step 2: Run Unified Training

```bash
python experiments/03_model_customization.py \
    --input ./results/unified_training_data/ \
    --config experiments/maveric_config.yaml \
    --output-dir ./results/unified_training/models \
    --unified-training
```

**Expected Output**:

```
⚙️  Training configuration:
   Epochs: 60
   Learning rate: 1e-07
   Weight decay: 0.05
   Optimizer: adamw
   Scheduler: cosine
   Skip epoch evaluation: True
   ⚠️  Per-epoch evaluation DISABLED (unified training mode)
   ✓  Checkpoints will be saved every 5 epoch(s)

🤖 Training unified model...

Epoch 1/60
Training: 100%|████████████| 235/235 [34:23<00:00]
Train Loss: 4.9121, Train Acc: 15.26% (evaluation skipped)
Saved checkpoint: checkpoint_epoch_1.pth

Epoch 2/60
Training: 100%|████████████| 235/235 [34:18<00:00]
Train Loss: 4.3706, Train Acc: 19.47% (evaluation skipped)

...

Epoch 5/60
Training: 100%|████████████| 235/235 [34:25<00:00]
Train Loss: 3.8124, Train Acc: 25.33% (evaluation skipped)
Saved checkpoint: checkpoint_epoch_5.pth
```

### Step 3: Evaluate Final Model Per-Dataset

```bash
python experiments/05_unified_evaluation.py \
    --checkpoint ./results/unified_training/models/checkpoint_epoch_60.pth \
    --config experiments/maveric_config.yaml
```

**Output** (per-dataset accuracy):

```
📊 UNIFIED MODEL EVALUATION RESULTS
══════════════════════════════════════════════════════════════════════════════
Dataset              Baseline  Customized  Δ (pp)    Samples
──────────────────────────────────────────────────────────────────────────────
Cars                   55.23      58.45     +3.22      8,041
CIFAR10                87.31      89.12     +1.81     10,000
CIFAR100               58.42      61.28     +2.86     10,000
...
```

## Time Savings

### Without `skip_epoch_evaluation` (old behavior)

```
Epoch 1: 35 min training + 7 min evaluation = 42 min
Epoch 2: 35 min training + 7 min evaluation = 42 min
...
Total for 60 epochs: 42 min × 60 = 42 hours
```

### With `skip_epoch_evaluation: true` (new behavior)

```
Epoch 1: 35 min training = 35 min
Epoch 2: 35 min training = 35 min
...
Total for 60 epochs: 35 min × 60 = 35 hours

Time saved: 7 hours (17% faster)
```

## Checkpoint Management

### When Evaluation is ENABLED (default)

- Saves only **best model** based on test accuracy
- Checkpoint name: `best_model.pth`
- Metadata includes: `epoch`, `test_acc`, `val_acc`, `is_best=True`
- Previous best checkpoint is deleted to save disk space

### When Evaluation is DISABLED (unified training)

- Saves periodic checkpoints every `save_frequency` epochs
- Checkpoint names: `checkpoint_epoch_1.pth`, `checkpoint_epoch_5.pth`, etc.
- Metadata includes: `epoch`, `train_acc`, `train_loss`, `is_best=False`
- All checkpoints are kept (no automatic deletion)
- **Recommendation**: Set `save_frequency` higher (e.g., 5-10) to save disk space

## Implementation Details

### Files Modified

1. **`maveric/config.py`** (line 295):
   - Added `skip_epoch_evaluation: bool = False` to `TrainingConfig` dataclass

2. **`maveric/customization/training.py`** (lines 134-222):
   - Modified `train()` method to check `training_config.skip_epoch_evaluation`
   - Added branch for skipping evaluation and saving periodic checkpoints

3. **`experiments/03_model_customization.py`** (line 541):
   - Added `skip_epoch_evaluation` parameter to `create_training_config()`

4. **`experiments/maveric_config.yaml`** (line 192):
   - Added `skip_epoch_evaluation: false` configuration option

### Code Flow

```python
# In Trainer.train() method

for epoch in range(training_config.epochs):
    # Always train
    train_loss, train_acc = self._train_epoch(...)

    # Check evaluation setting
    if not training_config.skip_epoch_evaluation and epoch % eval_frequency == 0:
        # Standard behavior: evaluate and save best model
        test_loss, test_acc = self._validate_epoch(...)
        if test_acc > best_val_acc:
            save_checkpoint("best_model")

    elif training_config.skip_epoch_evaluation:
        # Unified training: skip evaluation, save periodically
        log_info("(evaluation skipped)")
        if epoch % save_frequency == 0:
            save_checkpoint(f"checkpoint_epoch_{epoch+1}")
```

## Best Practices

### For Per-Dataset Training

```yaml
skip_epoch_evaluation: false  # Keep evaluation enabled
save_best_model: true
eval_frequency: 1
```

### For Unified Training

```yaml
skip_epoch_evaluation: true   # Disable evaluation
save_frequency: 5             # Save every 5-10 epochs
keep_last_n_checkpoints: 3   # Optional: auto-cleanup old checkpoints
```

### Checkpoint Selection Strategy

After unified training completes:

1. **Evaluate multiple checkpoints**:
   ```bash
   for epoch in 10 20 30 40 50 60; do
       python 05_unified_evaluation.py \
           --checkpoint models/checkpoint_epoch_${epoch}.pth \
           --config maveric_config.yaml
   done
   ```

2. **Compare per-dataset results** to find best checkpoint

3. **Use best checkpoint** for final evaluation/deployment

## FAQ

**Q: Why not just use `eval_frequency=0` to disable evaluation?**

A: The `eval_frequency` parameter controls how often evaluation runs (every N epochs), not whether it runs at all. Setting it to 0 would cause division errors. The `skip_epoch_evaluation` flag is a clearer, safer way to completely disable evaluation.

**Q: Can I use this for single-dataset training?**

A: Yes, but it's not recommended. For single-dataset training, per-epoch evaluation provides useful early stopping signals and helps select the best model checkpoint.

**Q: What happens to validation data when evaluation is skipped?**

A: Validation is also skipped (no need to waste time on it if test evaluation is skipped). Training uses 100% of data without validation split.

**Q: How do I know which checkpoint to use for final evaluation?**

A: Evaluate all saved checkpoints using `05_unified_evaluation.py` and compare per-dataset accuracies. Choose the checkpoint with best overall performance across datasets.

## Related Files

- Feature implementation: [maveric/customization/training.py](maveric/customization/training.py)
- Configuration: [maveric/config.py](maveric/config.py)
- Training script: [experiments/03_model_customization.py](experiments/03_model_customization.py)
- Evaluation script: [experiments/05_unified_evaluation.py](experiments/05_unified_evaluation.py)
- Config file: [experiments/maveric_config.yaml](experiments/maveric_config.yaml)
