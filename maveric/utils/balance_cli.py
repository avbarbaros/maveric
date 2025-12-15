#!/usr/bin/env python3
"""
Command-line interface for balancing manually cleaned training datasets.

This tool allows you to balance training JSON files after manual inspection
and cleanup, using the same strategies available in the interactive GUI.

IMPORTANT: This tool uses INTELLIGENT SAMPLE SELECTION based on consistency scores:
- Samples are sorted by 'consistency' score (higher = better quality)
- Undersampling: Keeps the TOP N samples with highest consistency
- Oversampling: Duplicates the best samples cyclically
- This ensures balanced datasets maintain the highest quality samples

This matches the behavior of maveric.visualization.interactive.apply_balance().
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def load_training_files(input_path: str) -> pd.DataFrame:
    """
    Load training JSON files (single file or directory with multiple files).

    Args:
        input_path: Path to single JSON file or directory containing JSON files

    Returns:
        DataFrame with all training samples
    """
    input_path = Path(input_path)
    all_samples = []

    if input_path.is_file():
        # Single file
        print(f"📂 Loading single file: {input_path}")
        with open(input_path, 'r') as f:
            samples = json.load(f)
            all_samples.extend(samples)
            print(f"   ✓ Loaded {len(samples)} samples")

    elif input_path.is_dir():
        # Directory with multiple files
        json_files = sorted(input_path.glob('*_training_maveric_*.json'))

        if not json_files:
            raise FileNotFoundError(f"No training JSON files found in {input_path}")

        print(f"📂 Loading {len(json_files)} training files from: {input_path}")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                samples = json.load(f)
                all_samples.extend(samples)
                print(f"   ✓ {json_file.name}: {len(samples)} samples")

    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    df = pd.DataFrame(all_samples)
    print(f"\n✅ Total loaded: {len(df)} samples")

    return df


def balance_dataset(df: pd.DataFrame,
                   strategy: str,
                   target_per_class: Optional[int] = None,
                   enable_undersampling: bool = True,
                   enable_oversampling: bool = False) -> pd.DataFrame:
    """
    Balance dataset using specified strategy with intelligent sample selection.

    This function matches the behavior of interactive.apply_balance():
    - Sorts samples by 'consistency' score (higher is better)
    - Undersampling: Keeps top N samples with highest consistency
    - Oversampling: Duplicates best samples cyclically
    - Final shuffle with fixed random seed for reproducibility

    Args:
        df: DataFrame with training samples (must have 'label' and 'consistency' columns)
        strategy: Balancing strategy ('min', 'max', 'mean', 'median', 'custom')
        target_per_class: Target samples per class (for 'custom' strategy)
        enable_undersampling: Whether to allow reducing samples
        enable_oversampling: Whether to allow duplicating samples

    Returns:
        Balanced DataFrame with samples sorted by quality
    """
    # Get class distribution
    class_counts = df['label'].value_counts()

    print(f"\n📊 Original class distribution:")
    for class_name, count in class_counts.sort_index().items():
        print(f"   {class_name}: {count} samples")

    # Determine target count per class
    if strategy == 'min':
        target_count = class_counts.min()
        print(f"\n🎯 Strategy: Minimum ({target_count} samples per class)")
    elif strategy == 'max':
        target_count = class_counts.max()
        print(f"\n🎯 Strategy: Maximum ({target_count} samples per class)")
    elif strategy == 'mean':
        target_count = int(class_counts.mean())
        print(f"\n🎯 Strategy: Mean ({target_count} samples per class)")
    elif strategy == 'median':
        target_count = int(class_counts.median())
        print(f"\n🎯 Strategy: Median ({target_count} samples per class)")
    elif strategy == 'custom':
        if target_per_class is None:
            raise ValueError("Must specify --target-per-class for custom strategy")
        target_count = target_per_class
        print(f"\n🎯 Strategy: Custom ({target_count} samples per class)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Balance each class
    balanced_samples = []

    print(f"\n⚖️  Balancing classes...")
    for class_name in sorted(df['label'].unique()):
        class_df = df[df['label'] == class_name].copy()
        current_count = len(class_df)

        # Sort by consistency score for intelligent sample selection
        # (same as interactive GUI - keeps best samples)
        if 'consistency' in class_df.columns:
            class_df = class_df.sort_values('consistency', ascending=False)
        else:
            print(f"   ⚠️  Warning: No 'consistency' column found, samples will not be sorted by quality")

        if current_count == target_count:
            # Already balanced
            balanced_samples.extend(class_df.to_dict('records'))
            print(f"   ✓ {class_name}: {current_count} → {target_count} (no change)")

        elif current_count > target_count:
            # Undersample (reduce) - take TOP samples sorted by consistency
            if enable_undersampling:
                # Take top N samples (highest consistency)
                sampled = class_df.head(target_count)
                balanced_samples.extend(sampled.to_dict('records'))
                print(f"   ↓ {class_name}: {current_count} → {target_count} (undersampled -{current_count - target_count}, kept best)")
            else:
                # Keep all samples
                balanced_samples.extend(class_df.to_dict('records'))
                print(f"   ⚠️ {class_name}: {current_count} → {current_count} (undersampling disabled)")

        else:  # current_count < target_count
            # Oversample (duplicate) - duplicate best samples cyclically
            if enable_oversampling:
                # Start with all samples
                sampled_list = class_df.to_dict('records')
                needed = target_count - current_count

                # Duplicate samples cyclically starting from best (highest consistency)
                for i in range(needed):
                    duplicate_idx = i % current_count
                    sampled_list.append(class_df.iloc[duplicate_idx].to_dict())

                balanced_samples.extend(sampled_list)
                print(f"   ↑ {class_name}: {current_count} → {target_count} (oversampled +{needed}, duplicated best)")
            else:
                # Keep all samples
                balanced_samples.extend(class_df.to_dict('records'))
                print(f"   ⚠️ {class_name}: {current_count} → {current_count} (oversampling disabled)")

    balanced_df = pd.DataFrame(balanced_samples)

    # Shuffle the data to randomize sample order
    # (same as interactive GUI - ensures balanced classes are mixed)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Show final distribution
    final_counts = balanced_df['label'].value_counts()
    print(f"\n📊 Balanced class distribution:")
    for class_name, count in final_counts.sort_index().items():
        print(f"   {class_name}: {count} samples")

    print(f"\n✅ Total balanced: {len(balanced_df)} samples (shuffled)")

    return balanced_df


def save_balanced_data(df: pd.DataFrame, output_path: str, rotation_size: int = 1000):
    """
    Save balanced dataset to JSON file(s).

    Args:
        df: Balanced DataFrame
        output_path: Output file or directory path
        rotation_size: Number of samples per file
    """
    output_path = Path(output_path)

    # Convert DataFrame to list of dicts
    samples = df.to_dict('records')
    total_samples = len(samples)

    if total_samples <= rotation_size:
        # Single file
        if output_path.is_dir():
            output_file = output_path / 'balanced_training_maveric_1.json'
        else:
            output_file = output_path

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"\n💾 Saved {total_samples} samples to: {output_file}")
        return str(output_file)

    else:
        # Multiple files
        if output_path.is_file():
            # User provided filename, use its parent directory
            output_dir = output_path.parent
            base_name = output_path.stem.replace('_1', '').replace('_training_maveric', '')
        else:
            output_dir = output_path
            base_name = 'balanced'

        output_dir.mkdir(parents=True, exist_ok=True)

        num_files = (total_samples + rotation_size - 1) // rotation_size

        print(f"\n💾 Saving {total_samples} samples to {num_files} files...")

        for file_idx in range(num_files):
            start_idx = file_idx * rotation_size
            end_idx = min((file_idx + 1) * rotation_size, total_samples)
            file_samples = samples[start_idx:end_idx]

            output_file = output_dir / f'{base_name}_training_maveric_{file_idx + 1}.json'

            with open(output_file, 'w') as f:
                json.dump(file_samples, f, indent=2)

            print(f"   ✓ {output_file.name}: {len(file_samples)} samples")

        print(f"\n✅ All files saved to: {output_dir}")
        return str(output_dir)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Balance manually cleaned training datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Balance using minimum class size (undersample majority classes)
  python -m maveric.utils.balance_cli --input ./curated_data --strategy min --output ./balanced_data

  # Balance to 500 samples per class
  python -m maveric.utils.balance_cli --input ./curated_data --strategy custom --target-per-class 500 --output ./balanced_data

  # Balance using mean, allow both under/oversampling
  python -m maveric.utils.balance_cli --input ./curated_data --strategy mean --enable-oversampling --output ./balanced_data

  # Balance single file
  python -m maveric.utils.balance_cli --input data.json --strategy median --output balanced.json

Strategies:
  min     : Use smallest class size (pure undersampling)
  max     : Use largest class size (pure oversampling)
  mean    : Use average class size (mixed sampling)
  median  : Use median class size (mixed sampling)
  custom  : Use specified target (with --target-per-class)
        """
    )

    # Input/output
    parser.add_argument('--input', '-i', required=True,
                       help='Input JSON file or directory with training files')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSON file or directory')

    # Balancing strategy
    parser.add_argument('--strategy', '-s',
                       choices=['min', 'max', 'mean', 'median', 'custom'],
                       default='min',
                       help='Balancing strategy (default: min)')
    parser.add_argument('--target-per-class', '-t', type=int,
                       help='Target samples per class (required for custom strategy)')

    # Sampling options
    parser.add_argument('--enable-undersampling', action='store_true', default=True,
                       help='Allow reducing samples (default: True)')
    parser.add_argument('--disable-undersampling', dest='enable_undersampling',
                       action='store_false',
                       help='Disable undersampling')
    parser.add_argument('--enable-oversampling', action='store_true', default=False,
                       help='Allow duplicating samples (default: False)')

    # File options
    parser.add_argument('--rotation-size', type=int, default=1000,
                       help='Samples per output file (default: 1000)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without saving')

    args = parser.parse_args()

    try:
        print("="*80)
        print("MAVERIC Dataset Balancing Tool")
        print("="*80)

        # Load data
        df = load_training_files(args.input)

        # Balance data
        balanced_df = balance_dataset(
            df,
            strategy=args.strategy,
            target_per_class=args.target_per_class,
            enable_undersampling=args.enable_undersampling,
            enable_oversampling=args.enable_oversampling
        )

        # Save or show dry run
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No files saved")
            print(f"   Would save to: {args.output}")
        else:
            save_balanced_data(balanced_df, args.output, args.rotation_size)

        print("\n" + "="*80)
        print("✅ Balancing complete!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
