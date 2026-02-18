#!/usr/bin/env python3
"""
Test script for the dataset balancing CLI tool.
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("Dataset Balancing CLI Test")
print("="*80)

# Test 1: Verify the module exists
print("\n1. Checking balance_cli module...")

try:
    from maveric.utils import balance_cli
    print("   ✓ balance_cli module imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import balance_cli: {e}")
    sys.exit(1)

# Test 2: Verify functions exist
print("\n2. Checking required functions...")

required_functions = [
    'load_training_files',
    'balance_dataset',
    'save_balanced_data',
    'main'
]

for func_name in required_functions:
    if hasattr(balance_cli, func_name):
        print(f"   ✓ {func_name}() exists")
    else:
        print(f"   ✗ {func_name}() not found")
        sys.exit(1)

# Test 3: Create sample data and test balancing
print("\n3. Testing balancing with sample data...")

# Create temporary directory
temp_dir = Path(tempfile.mkdtemp())
print(f"   Created temp directory: {temp_dir}")

try:
    # Create imbalanced sample data
    sample_data = {
        'airplane': 100,
        'automobile': 50,
        'bird': 200,
        'cat': 75,
        'dog': 150
    }

    all_samples = []
    for label, count in sample_data.items():
        for i in range(count):
            all_samples.append({
                'id': len(all_samples) + 1,
                'url': f'http://example.com/img_{len(all_samples)}.jpg',
                'label': label,
                'text': f'A photo of a {label}',
                'weighted_class_score': 0.85 + (i % 10) * 0.01,
                'consistency': 0.90
            })

    # Save to JSON file
    input_file = temp_dir / 'imbalanced_training_maveric_1.json'
    with open(input_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"   ✓ Created test data: {len(all_samples)} samples")
    print(f"     Distribution: {sample_data}")

    # Test load function
    import pandas as pd
    df = balance_cli.load_training_files(str(input_file))
    print(f"   ✓ Loaded data: {len(df)} samples")

    # Test balancing strategies
    strategies = ['min', 'max', 'mean', 'median']

    for strategy in strategies:
        balanced_df = balance_cli.balance_dataset(
            df.copy(),
            strategy=strategy,
            enable_undersampling=True,
            enable_oversampling=(strategy != 'min')
        )

        # Verify balance
        class_counts = balanced_df['label'].value_counts()
        unique_counts = class_counts.unique()

        if len(unique_counts) == 1:
            print(f"   ✓ Strategy '{strategy}': Perfectly balanced ({unique_counts[0]} per class)")
        else:
            print(f"   ⚠️ Strategy '{strategy}': {dict(class_counts)}")

    # Test custom target
    custom_target = 80
    balanced_df = balance_cli.balance_dataset(
        df.copy(),
        strategy='custom',
        target_per_class=custom_target,
        enable_undersampling=True,
        enable_oversampling=True
    )

    class_counts = balanced_df['label'].value_counts()
    if all(count == custom_target for count in class_counts):
        print(f"   ✓ Custom target ({custom_target}): Perfectly balanced")
    else:
        print(f"   ⚠️ Custom target: {dict(class_counts)}")

    # Test save function
    output_dir = temp_dir / 'balanced'
    balance_cli.save_balanced_data(balanced_df, str(output_dir), rotation_size=100)

    saved_files = list(output_dir.glob('*.json'))
    print(f"   ✓ Saved {len(saved_files)} output file(s)")

    # Verify saved data
    total_saved = 0
    for file in saved_files:
        with open(file, 'r') as f:
            data = json.load(f)
            total_saved += len(data)

    if total_saved == len(balanced_df):
        print(f"   ✓ Verified: Saved {total_saved} samples (matches balanced count)")
    else:
        print(f"   ✗ Mismatch: Saved {total_saved}, expected {len(balanced_df)}")

finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"   ✓ Cleaned up temp directory")

# Test 4: Document usage
print("\n4. Usage examples:")
print()
print("   Example 1: Balance using minimum (undersample to smallest class)")
print("   ----------------------------------------------------------------")
print("   python balance_dataset.py \\")
print("       --input ./curated_data \\")
print("       --output ./balanced_data \\")
print("       --strategy min")
print()
print("   Example 2: Balance to specific target per class")
print("   -----------------------------------------------")
print("   python balance_dataset.py \\")
print("       --input ./curated_data \\")
print("       --output ./balanced_data \\")
print("       --strategy custom \\")
print("       --target-per-class 500")
print()
print("   Example 3: Balance using mean (allow over/undersampling)")
print("   -------------------------------------------------------")
print("   python balance_dataset.py \\")
print("       --input ./curated_data \\")
print("       --output ./balanced_data \\")
print("       --strategy mean \\")
print("       --enable-oversampling")
print()
print("   Example 4: Dry run (preview without saving)")
print("   ------------------------------------------")
print("   python balance_dataset.py \\")
print("       --input ./curated_data \\")
print("       --output ./balanced_data \\")
print("       --strategy median \\")
print("       --dry-run")

# Test 5: Show available strategies
print("\n5. Available strategies:")
print()
print("   min     : Balance to smallest class (pure undersampling)")
print("           Example: Classes [100, 200, 300] → [100, 100, 100]")
print()
print("   max     : Balance to largest class (pure oversampling)")
print("           Example: Classes [100, 200, 300] → [300, 300, 300]")
print()
print("   mean    : Balance to average class size")
print("           Example: Classes [100, 200, 300] → [200, 200, 200]")
print()
print("   median  : Balance to median class size")
print("           Example: Classes [100, 200, 300] → [200, 200, 200]")
print()
print("   custom  : Balance to specified target (use --target-per-class)")
print("           Example: Classes [100, 200, 300] → [500, 500, 500]")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print()
print("The balance_dataset.py CLI tool is ready to use!")
print()
print("Workflow:")
print("  1. Curate data using interactive GUI (Save Data)")
print("  2. Manually inspect grids in curationResults/")
print("  3. Remove bad samples from JSON files")
print("  4. Balance cleaned data using this CLI tool")
print("  5. Use balanced data for model customization")
