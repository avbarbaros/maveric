# Global vs Per-Class Filtering - Visual Explanation

## Overview

The Mahalanobis Filter tab offers two filtering modes: **Global** and **Per-Class**. Understanding the difference is crucial for getting the results you want.

---

## Global Mode

**How it works**: Applies Mahalanobis filtering to the **entire dataset** as a single population.

### Example Scenario

**Before filtering** (50,000 samples, 10 classes):
```
airplane:     5,000 samples
automobile:   5,000 samples
bird:         5,000 samples
cat:          5,000 samples
deer:         5,000 samples
dog:          5,000 samples
frog:         5,000 samples
horse:        5,000 samples
ship:         5,000 samples
truck:        5,000 samples
```

**After Global filtering at 30%** (15,000 samples total):
```
airplane:       900 samples  ← Lost 4,100 (82%)
automobile:   1,200 samples  ← Lost 3,800 (76%)
bird:         2,100 samples  ← Lost 2,900 (58%)  ← More samples kept!
cat:          1,800 samples  ← Lost 3,200 (64%)
deer:         1,500 samples  ← Lost 3,500 (70%)
dog:          1,900 samples  ← Lost 3,100 (62%)
frog:         1,600 samples  ← Lost 3,400 (68%)
horse:        1,400 samples  ← Lost 3,600 (72%)
ship:         1,300 samples  ← Lost 3,700 (74%)
truck:        1,300 samples  ← Lost 3,700 (74%)
```

### Key Characteristics

✅ **Advantage**: Keeps the **absolute best** samples across all classes
✅ **Advantage**: Maximizes overall quality
❌ **Disadvantage**: Can create **class imbalance**
❌ **Disadvantage**: Some classes may have very few samples (like "airplane" with only 900)

### Why Imbalance Occurs

Different classes have different quality distributions:
- **Bird**: Might have many high-quality samples (good photos, clear captions)
- **Airplane**: Might have fewer high-quality samples (motion blur, complex backgrounds)

Global mode doesn't care about class—it just picks the top 30% best samples overall, which may favor certain classes.

### Visual Example

```
Quality Space (weighted_score vs consistency)

                  Ideal Point (★)
                       ↓
    High    ╔═════════════════╗
    Cons.   ║    ● bird       ║  ← Many bird samples in high-quality region
            ║  ● bird         ║
            ║   ● dog         ║
            ║     ● cat       ║
            ║      ● bird     ║
    ────────╫─────────────────╫─── Selection boundary (ellipse)
            ║                 ║
    Low     ║ ● airplane      ║  ← Few airplane samples in high-quality region
    Cons.   ╚═════════════════╝
            Low              High
           Weighted Score
```

---

## Per-Class Mode

**How it works**: Applies Mahalanobis filtering **separately for each class**, ensuring each class contributes proportionally.

### Example Scenario

**Before filtering** (50,000 samples, 10 classes):
```
airplane:     5,000 samples
automobile:   5,000 samples
bird:         5,000 samples
cat:          5,000 samples
deer:         5,000 samples
dog:          5,000 samples
frog:         5,000 samples
horse:        5,000 samples
ship:         5,000 samples
truck:        5,000 samples
```

**After Per-Class filtering at 30%** (15,000 samples total):
```
airplane:     1,500 samples  ← Exactly 30% of original (balanced!)
automobile:   1,500 samples  ← Exactly 30% of original
bird:         1,500 samples  ← Exactly 30% of original
cat:          1,500 samples  ← Exactly 30% of original
deer:         1,500 samples  ← Exactly 30% of original
dog:          1,500 samples  ← Exactly 30% of original
frog:         1,500 samples  ← Exactly 30% of original
horse:        1,500 samples  ← Exactly 30% of original
ship:         1,500 samples  ← Exactly 30% of original
truck:        1,500 samples  ← Exactly 30% of original
```

### Key Characteristics

✅ **Advantage**: Maintains **perfect class balance**
✅ **Advantage**: Every class gets equal representation
✅ **Advantage**: Better for training balanced models
❌ **Disadvantage**: Might keep some **lower-quality samples** from classes with poor overall quality
❌ **Disadvantage**: Slightly lower overall quality than Global mode

### How It Works

For each class independently:
1. Calculate ideal point for **that class only**
2. Compute Mahalanobis distances for **that class only**
3. Keep top 30% of **that class**
4. Combine all classes

### Visual Example

```
Per-Class Processing (each class processed separately)

Airplane Class:              Bird Class:
    High    ╔═══════╗           High    ╔═══════╗
    Cons.   ║  ★    ║           Cons.   ║  ★    ║
            ║ ●  ●  ║                   ║ ● ● ● ║  ← More samples overall
    ────────╫───────╫───        ────────╫───────╫───
            ║ ● ●   ║                   ║ ● ●   ║
    Low     ╚═══════╝           Low     ╚═══════╝

Keep top 30% from        Keep top 30% from
airplane (1,500)         bird (1,500)

Result: Both classes equally represented!
```

---

## Comparison Table

| Aspect | Global | Per-Class |
|--------|--------|-----------|
| **Selection Strategy** | Best samples overall | Best samples per class |
| **Class Balance** | May be imbalanced | Always balanced |
| **Overall Quality** | Higher (best of best) | Slightly lower (includes some weaker classes) |
| **Use Case** | When quality > balance | When balance is critical |
| **Result Example (30%)** | 15K samples, imbalanced | 15K samples, perfectly balanced |
| **Training Impact** | May favor certain classes | Equal learning for all classes |

---

## When to Use Each Mode

### Use **Global** When:

1. **Quality is paramount**: You want absolutely the best samples, regardless of class
2. **Class imbalance is okay**: Your model can handle imbalanced training data
3. **Some classes are clearly better**: You know certain classes have higher-quality data
4. **Post-balancing available**: You'll balance classes in Tab 4 afterward

**Example Use Case**:
```
"I want the highest quality 10,000 samples for fine-tuning CLIP.
I don't care if some classes have more samples than others."

→ Use Global mode with 20%
```

### Use **Per-Class** When:

1. **Balance is critical**: Your model needs equal representation from all classes
2. **Fair evaluation**: You want unbiased performance across all classes
3. **All classes matter**: Every class is equally important for your task
4. **Direct training**: You'll use filtered data directly without further balancing

**Example Use Case**:
```
"I'm training a classifier and need 1,500 samples per class
to ensure balanced learning across all 10 CIFAR-10 classes."

→ Use Per-Class mode with 30%
```

---

## Practical Examples

### Example 1: High-Quality Dataset (Use Global)

**Goal**: Create a premium dataset of 5,000 samples for few-shot learning

**Settings**:
- Mode: **Global**
- Keep Top: **10%** (from 50,000 → 5,000)

**Result**:
- 5,000 absolutely best samples
- Imbalanced (e.g., 800 bird, 300 airplane)
- Highest overall quality
- **Good for**: Transfer learning, few-shot learning

---

### Example 2: Balanced Training Set (Use Per-Class)

**Goal**: Create a balanced dataset of 15,000 samples for classifier training

**Settings**:
- Mode: **Per-Class**
- Keep Top: **30%** (from 50,000 → 15,000)

**Result**:
- 1,500 samples per class (perfectly balanced)
- Good quality for each class
- **Good for**: Training classifiers, evaluation benchmarks

---

### Example 3: Hybrid Approach (Global → Balance)

**Goal**: Get best quality with eventual balance

**Steps**:
1. **Tab 2**: Global mode, 40% (50K → 20K)
2. **Tab 4**: Balance to 1,500 per class (20K → 15K)

**Result**:
- High-quality samples (from Global filtering)
- Balanced classes (from Tab 4)
- **Best of both worlds**

---

## Common Mistakes

### ❌ Mistake 1: Using Global When Balance Is Critical

```
Scenario: Training a 10-class classifier
User selects: Global mode, 30%
Result: Imbalanced dataset (300-2,100 samples per class)
Problem: Model learns some classes better than others
Solution: Use Per-Class mode instead
```

### ❌ Mistake 2: Using Per-Class When Quality Matters Most

```
Scenario: Creating a demo dataset of "best examples"
User selects: Per-Class mode, 10%
Result: Includes mediocre samples from weak classes
Problem: Some samples don't meet quality standards
Solution: Use Global mode instead
```

### ❌ Mistake 3: Not Applying Tab 1 First

```
Scenario: User jumps directly to Tab 2
User clicks: Apply Filter
Result: Error - "weighted_class_score column not found"
Problem: Required columns not created yet
Solution: Go to Tab 1, click "Apply Settings" first
```

---

## Technical Details

### Global Mode Algorithm

```python
# Compute ideal point from ALL samples
ideal = [95th_percentile(all_weighted_scores),
         95th_percentile(all_consistency)]

# Compute covariance from ALL samples
covariance = cov(all_samples)

# Calculate distance for each sample
for sample in all_samples:
    distance = mahalanobis(sample, ideal, covariance)

# Sort by distance and keep top N%
keep_threshold = percentile(distances, N)
selected = samples[distance <= keep_threshold]
```

### Per-Class Mode Algorithm

```python
# For EACH class separately
for class_name in classes:
    class_samples = samples[label == class_name]

    # Compute ideal point for THIS CLASS
    ideal = [95th_percentile(class_weighted_scores),
             95th_percentile(class_consistency)]

    # Compute covariance for THIS CLASS
    covariance = cov(class_samples)

    # Calculate distance for samples in THIS CLASS
    for sample in class_samples:
        distance = mahalanobis(sample, ideal, covariance)

    # Keep top N% of THIS CLASS
    keep_threshold = percentile(class_distances, N)
    class_selected = class_samples[distance <= keep_threshold]

    selected.append(class_selected)

# Combine all classes
final = concatenate(selected)
```

---

## Summary

- **Global**: Best overall quality, may be imbalanced
- **Per-Class**: Guaranteed balance, slightly lower quality

Choose based on your priorities:
- **Quality > Balance**: Use Global
- **Balance > Quality**: Use Per-Class
- **Both**: Use Global, then balance in Tab 4

---

## Automatic Reset Feature

When you change the percentage and click "Apply Filter" again, MAVERIC **automatically resets** to the data before the previous Mahalanobis filter:

```
Initial state (after Tab 1):
   50,000 samples

Click Apply (30%):
   ✅ Filters to 15,000 samples
   💾 Backs up the 50,000 samples

Change to 20%, Click Apply:
   🔄 Resets to 50,000 samples
   ✅ Filters to 10,000 samples (20% of 50,000)
   ❌ Does NOT filter 20% of 15,000 = 3,000

This prevents compounding filters!
```

---

**Tip**: Try both modes and compare the statistics! The GUI shows you exactly how many samples per class are kept, so you can make an informed decision.
