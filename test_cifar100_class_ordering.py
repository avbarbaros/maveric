#!/usr/bin/env python3
"""
Test script to verify CIFAR-100 class name ordering matches torchvision.

This test ensures that ELEVATER class names align with torchvision's CIFAR-100
dataset to prevent label mismatches during evaluation.

Expected result:
- All 100 class names should match torchvision ordering (with spaces instead of underscores)
- No class name mismatches should occur
"""

import sys


def test_cifar100_ordering():
    """Test that CIFAR-100 class names match torchvision ordering."""
    print("=" * 80)
    print("CIFAR-100 Class Name Ordering Test")
    print("=" * 80)

    # Import ELEVATER dataset definition
    from maveric.datasets.elevater_datasets import ELEVATERDataset

    # Torchvision's CIFAR-100 class names (alphabetically sorted with underscores)
    torchvision_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
        'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
        'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
        'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
        'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
        'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    # Get ELEVATER CIFAR-100 class names
    elevater_classes = ELEVATERDataset.ELEVATER_DATASETS['cifar100']['class_names']

    print(f"\n1. Checking class count...")
    print(f"   Torchvision classes: {len(torchvision_classes)}")
    print(f"   ELEVATER classes: {len(elevater_classes)}")

    if len(torchvision_classes) != len(elevater_classes):
        print(f"   ❌ FAILED: Class count mismatch!")
        return False
    print(f"   ✓ Both have 100 classes")

    print(f"\n2. Checking class name ordering...")
    mismatches = []
    for i, (tv, el) in enumerate(zip(torchvision_classes, elevater_classes)):
        # Convert torchvision underscore to space for comparison
        tv_normalized = tv.replace('_', ' ')
        if tv_normalized != el:
            mismatches.append({
                'index': i,
                'torchvision': tv,
                'torchvision_normalized': tv_normalized,
                'elevater': el
            })

    if mismatches:
        print(f"   ❌ FAILED: Found {len(mismatches)} mismatches!")
        print(f"\n   First 10 mismatches:")
        for m in mismatches[:10]:
            print(f"      Index {m['index']}: torchvision='{m['torchvision']}' ({m['torchvision_normalized']}) vs ELEVATER='{m['elevater']}'")
        return False

    print(f"   ✓ All 100 class names match torchvision ordering")

    print(f"\n3. Verifying critical class positions...")
    critical_checks = [
        (0, 'apple'),
        (1, 'aquarium fish'),
        (21, 'chimpanzee'),
        (41, 'lawn mower'),
        (47, 'maple tree'),
        (52, 'oak tree'),
        (56, 'palm tree'),
        (58, 'pickup truck'),
        (83, 'sweet pepper'),
        (91, 'trout'),
        (92, 'tulip'),
        (96, 'willow tree'),
        (99, 'worm')
    ]

    for idx, expected in critical_checks:
        actual = elevater_classes[idx]
        if actual == expected:
            print(f"   ✓ Index {idx}: '{expected}' (correct)")
        else:
            print(f"   ❌ Index {idx}: expected '{expected}', got '{actual}'")
            return False

    print(f"\n4. Displaying class ranges...")
    print(f"   First 10 classes: {elevater_classes[:10]}")
    print(f"   Last 10 classes: {elevater_classes[-10:]}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nCIFAR-100 class names are correctly aligned with torchvision ordering.")
    print("Baseline model evaluation should now show ~65% accuracy instead of ~1%.")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = test_cifar100_ordering()
    sys.exit(0 if success else 1)
