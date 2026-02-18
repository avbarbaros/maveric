# Class-Organized Grid Visualization Example

## How Images Are Arranged

### CIFAR-10 Example (5,250 samples, 525 per class)

When you save grids, images are **sorted by class label** before creating the grids. This groups all images of the same class together.

### Grid Layout (10x10 = 100 images per grid)

#### Grid 1 (Samples 1-100): **airplane** class
```
┌──────────────────────────────────────────────────────────────┐
│  CIFAR10 Curation Results - Grid 1/53                       │
│  Samples 1-100 (Total: 5,250)                               │
│  Classes: airplane                                           │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│ ID:1│ ID:2│ ID:3│ ID:4│ ID:5│ ID:6│ ID:7│ ID:8│ ID:9│ID:10│
│ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │
│plane│plane│plane│plane│plane│plane│plane│plane│plane│plane│
│S:.92│S:.88│S:.90│S:.91│S:.87│S:.93│S:.89│S:.95│S:.90│S:.88│
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ID:11│ID:12│ID:13│ID:14│ID:15│ID:16│ID:17│ID:18│ID:19│ID:20│
│ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │
│plane│plane│plane│plane│plane│plane│plane│plane│plane│plane│
│ ... │ ... │ ... │ ... │ ... │ ... │ ... │ ... │ ... │ ... │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
All 100 images: airplane ✈️
```

#### Grid 2 (Samples 101-200): **airplane** class (continued)
```
┌──────────────────────────────────────────────────────────────┐
│  CIFAR10 Curation Results - Grid 2/53                       │
│  Samples 101-200 (Total: 5,250)                             │
│  Classes: airplane                                           │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│ID:101│ID:102│ ... │     │     │     │     │     │     │ID:200│
│ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │ ✈️  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
All 100 images: airplane ✈️
```

#### Grid 3-5: **airplane** (continued, 325 more)
...

#### Grid 6 (Samples 501-600): **airplane** ends, **automobile** starts
```
┌──────────────────────────────────────────────────────────────┐
│  CIFAR10 Curation Results - Grid 6/53                       │
│  Samples 501-600 (Total: 5,250)                             │
│  Classes: airplane, automobile (2 classes)                   │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│ID:501│ID:502│ ... │ID:525│ID:526│ID:527│ ... │     │     │ID:600│
│ ✈️  │ ✈️  │ ✈️  │ ✈️  │ 🚗  │ 🚗  │ 🚗  │ 🚗  │ 🚗  │ 🚗  │
│plane│plane│plane│plane│ car │ car │ car │ car │ car │ car │
│ ← Last 25 airplanes → │ ← First 75 automobiles →         │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
Mixed: airplane (25) + automobile (75)
```

#### Grid 7-10: **automobile** 🚗
...

#### Grid 11: **automobile** ends, **bird** starts
...

And so on for all 10 classes.

## Class Distribution Across Grids

For CIFAR-10 with 525 samples per class:

| Grid # | Samples     | Classes              | Description                    |
|--------|-------------|----------------------|--------------------------------|
| 1-5    | 1-500       | airplane ✈️          | All airplane                   |
| 6      | 501-600     | airplane + automobile| 25 airplane + 75 automobile    |
| 7-10   | 601-1000    | automobile 🚗        | All automobile                 |
| 11     | 1001-1100   | automobile + bird    | 50 automobile + 50 bird        |
| 12-15  | 1101-1500   | bird 🐦              | All bird                       |
| 16     | 1501-1600   | bird + cat           | 75 bird + 25 cat               |
| ...    | ...         | ...                  | ...                            |
| 53     | 5201-5250   | truck 🚚             | Last 50 trucks                 |

## Benefits of Class Organization

✅ **Easy Class Inspection**: See all images of one class together
✅ **Quality Comparison**: Compare quality within a class
✅ **Class Balance Check**: Visually verify each class has enough samples
✅ **Quick Navigation**: Know which grid contains which classes
✅ **Error Detection**: Mislabeled images stand out when surrounded by correct class

## Example Console Output

```
📊 Creating 53 grid visualization(s) for 5,250 samples...
   Grid size: 10x10 (100 images per grid)
   Organization: Class by class (sorted by label)
   Class distribution: {
     'airplane': 525, 'automobile': 525, 'bird': 525, 'cat': 525,
     'deer': 525, 'dog': 525, 'frog': 525, 'horse': 525,
     'ship': 525, 'truck': 525
   }
   Source: /path/to/results/images
   Output: /path/to/results/curationResults
   ✓ Saved grid 1/53: cifar10_grid_001.png    [airplane]
   ✓ Saved grid 2/53: cifar10_grid_002.png    [airplane]
   ...
   ✓ Saved grid 6/53: cifar10_grid_006.png    [airplane→automobile]
   ...
   ✓ Saved grid 53/53: cifar10_grid_053.png   [truck]

✅ All grids saved to: /path/to/results/curationResults
```

## How to Navigate Grids

1. **Find a specific class**: Check the class distribution in console output
2. **Example**: Want to inspect "cat" images?
   - cats: 525 samples (ranks 4th alphabetically)
   - First 3 classes: airplane (525) + automobile (525) + bird (525) = 1,575
   - cats start at sample 1,576
   - Grid number: 1,576 ÷ 100 = Grid 16 (starting from row 8)

3. **Grid title shows classes**: Each grid title lists which classes it contains
