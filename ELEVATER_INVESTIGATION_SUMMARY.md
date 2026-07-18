# ELEVATER Configuration Investigation

## Summary of Findings

Investigated ELEVATER's VOC2007 implementation from: https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC

### Key Findings

#### 1. **Annotation Handling** ✅ SAME AS OURS (post-fix)

**ELEVATER Code** (`dataset.py` lines 57-68):
```python
flag = 1
if line[7:9] and int(line[7:9]) != 1:
    flag = -1
if flag == 1:
    labels_all[index][label_int] = 1
```

**Interpretation**:
- If annotation flag == 1: label = 1 (present, not difficult)
- If annotation flag == 0 or -1: label = 0 (absent or difficult)

**They treat difficult examples (flag=0) as negatives (label=0)**, same as we did **before** our difficult-example protocol fix.

**Implication**: Our difficult-example fix (excluding difficult from FP count) might have been incorrect or unnecessary, which explains why it only improved mAP by 0.55%.

#### 2. **Text Templates** ✅ IDENTICAL

**ELEVATER** (`prompts.py`):
```python
pascalvoc2007_templates = [
    'a photo of a {}.',
]
```

**MAVERIC**: Same template

✅ **No difference here**

#### 3. **Class Names** ✅ IDENTICAL

Both use:
```python
['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

✅ **No difference here**

#### 4. **mAP Metric Implementation** ⚠️ DIFFERENT LIBRARY

**ELEVATER** (`metric.py`):
```python
def map_11_points(y_label, y_pred_proba):
    evaluator = v_eval.MeanAveragePrecisionNPointsEvaluator(11)
    evaluator.add_predictions(predictions=y_pred_proba, targets=y_label)
    return evaluator.get_report()[evaluator._get_id()]
```

They use an external library: `vision_evaluation.evaluators.MeanAveragePrecisionNPointsEvaluator(11)`

**MAVERIC**: Custom implementation in `_compute_voc11_map()`

⚠️ **Potential difference**: Implementation details in the library vs our custom code

#### 5. **Score Processing** ⚠️ POTENTIALLY DIFFERENT

**ELEVATER** (`clip_zeroshot_evaluator.py` line 19):
```python
logits = (100. * image_features @ text_features).softmax(dim=-1)
result = metric(image_labels.squeeze().cpu().detach().numpy(), logits.cpu().detach().numpy())
```

They:
1. Compute similarity: `image_features @ text_features`
2. Scale by 100: `100. * ...`
3. Apply **softmax**: `.softmax(dim=-1)`
4. Pass softmax probabilities to mAP metric

**MAVERIC**: We pass **raw logits** (similarity scores) to mAP

⚠️ **Potential difference**: Softmax probabilities vs raw scores

#### 6. **Image Preprocessing** ❓ UNKNOWN

Could not find explicit preprocessing/transforms in the code examined. Key unknowns:
- Image resize strategy (shortest edge? center crop?)
- Normalization values
- Aspect ratio handling

## Current MAVERIC Results

- **Original (broken)**: 61.36% mAP
- **Multi-label fix**: 75.07% mAP (+13.71 points)
- **+ Difficult-example protocol**: 75.62% mAP (+0.55 points)
- **Target (ELEVATER reported)**: 82.60% mAP
- **Remaining gap**: 7.0 percentage points

## Hypotheses for Remaining Gap

### Hypothesis 1: Score Processing (MOST LIKELY)

ELEVATER uses softmax probabilities while we use raw scores.

**Test**: Modify our evaluation to apply softmax before computing mAP:
```python
# In evaluation.py, before passing to mAP:
all_scores = torch.softmax(torch.from_numpy(all_scores * 100), dim=-1).numpy()
```

**Expected impact**: Could explain part or all of the 7% gap

### Hypothesis 2: External Library Differences

ELEVATER uses `vision_evaluation` library's mAP implementation, we use custom code.

**Test**: Install and use the same library, or carefully verify our implementation matches their algorithm.

### Hypothesis 3: Image Preprocessing

Unknown differences in:
- Image resizing strategy
- Normalization constants  
- Aspect ratio handling

**Test**: Extract and match exact preprocessing from ELEVATER's code.

### Hypothesis 4: Baseline Verification

The 82.60% might be:
- From a different CLIP model (not ViT-B/32)
- After fine-tuning (not zero-shot)
- Using ensemble of templates (not single template)
- From a different evaluation protocol

**Test**: Find ELEVATER's published results table to verify the exact configuration.

## Recommendations

### Immediate Actions

1. **Apply softmax to scores** (Hypothesis 1):
   ```python
   # In _compute_voc11_map or before calling it:
   scores = torch.softmax(torch.from_numpy(scores * 100), dim=-1).numpy()
   ```

2. **Verify difficult-example handling**:
   Since ELEVATER treats difficult as negatives, we should **revert** our difficult-example protocol fix and use the simpler approach (difficult = negative).

3. **Check preprocessing**:
   Compare CLIP processor settings with ELEVATER's transforms.

### Long-term

1. **Find ELEVATER's published baseline table** to verify 82.60% is for:
   - Zero-shot CLIP ViT-B/32
   - Single template
   - VOC2007 test set

2. **Install vision_evaluation library** and use their exact mAP implementation for direct comparison.

## Files from ELEVATER Repository

- `vision_benchmark/evaluation/dataset.py` - VOC2007 dataset class
- `vision_benchmark/evaluation/metric.py` - mAP metric wrapper
- `vision_benchmark/evaluation/clip_zeroshot_evaluator.py` - Evaluation logic with softmax
- `vision_benchmark/datasets/prompts.py` - Text templates
- `vision_benchmark/models/clip_react.py` - CLIP model implementation

## Next Steps

**Priority 1**: Test softmax hypothesis
**Priority 2**: Verify baseline number from ELEVATER paper/results
**Priority 3**: Match preprocessing exactly

The softmax hypothesis is most promising because:
- It's a clear, concrete difference
- It affects score distributions which could significantly impact ranking
- Easy to test immediately
